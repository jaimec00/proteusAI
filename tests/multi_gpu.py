import torch
from concurrent.futures import ThreadPoolExecutor


def main():
    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")

    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

    results = []
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(test, i, devices[i])
            for i in range(num_gpus)
        ]
        for future in futures:
            results.append(future.result())

    print(results, [result.device for result in results ])

def test(num, device):
    x = torch.ones(num).to(device)
    x += 44
    return x

if __name__ == '__main__':
    main()