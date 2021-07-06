from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import average_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results_v2
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from client import Client

def train_subset_of_clients(epoch, args, clients, poisoned_workers):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)

    for client_idx in random_workers:
        args.get_logger().info("Đang train epoch #{} ở client #{}", str(epoch), str(clients[client_idx].get_client_index()))
        clients[client_idx].train(epoch)

    args.get_logger().info("Đang tính trung bình các parameter từ client.")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    new_nn_params = average_nn_parameters(parameters)

    for client in clients:
        client.update_nn_parameters(new_nn_params)
        args.get_logger().info("Đang cập nhật tham số cho client #{}", str(client.get_client_index()))

    return clients[0].test(), random_workers

def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))

    return clients

def run_machine_learning(clients, args, poisoned_workers):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    worker_selection = []
    for epoch in range(1, args.get_num_epochs() + 1):
        results, workers_selected = train_subset_of_clients(epoch, args, clients, poisoned_workers)

        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)

    return convert_results_to_csv(epoch_test_set_results), worker_selection

def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx):
    """
    Hàm chạy thực nghiệm tấn công
    """
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)
    """ Lấy đường dẫn lưu log, kết quả, model, danh sách work được chọn"""

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)
    """Nạp các cài đặt"""
    args = Arguments(logger)
    args.set_model_save_path(models_folders[0]) # cài đặt đường dẫn lưu model
    args.set_num_poisoned_workers(num_poisoned_workers) # nạp số lượng worker có poison data
    args.set_round_worker_selection_strategy_kwargs(KWARGS) # nạp số lượng work mỗi round
    args.set_client_selection_strategy(client_selection_strategy) # hàm chọn worker
    args.log() # log lại

    train_data_loader = load_train_data_loader(logger, args) # tạo train_data_loader từ dữ liệu
    test_data_loader = load_test_data_loader(logger, args) # tạo test_data_loader từ dữ liệu

    # Distribute batches equal volume IID
    """Chia dữ liệu cho worker giống nhau"""
    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())

    """chuyển dữ liệu về mảng numpy"""
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)

    """Chọn ngẫu nhiên các một số lượng worker trong tổng số worker"""
    poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())

    """
    Nhiễm độc train dataset cho các worker với một replacement method nhất định
    """
    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers, replacement_method)

    """tạo các train data loader mới"""
    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    """Tạo các client"""
    clients = create_clients(args, train_data_loaders, test_data_loader)

    results, worker_selection = run_machine_learning(clients, args, poisoned_workers)
    save_results_v2(results, results_files[0])
    save_results_v2(worker_selection, worker_selections_files[0])

    logger.remove(handler)
