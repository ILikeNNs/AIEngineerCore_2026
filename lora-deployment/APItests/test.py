import requests


def main():
    url = "https://korona-lukasz--vllm-lora-service-api.modal.run"

    payload = {"prompt": "In the acute phase of HCV infection, what enzyme undergoes a rise and fall within a period of six months?"}

    # Use requests.get() if it is a GET endpoint
    response = requests.post(url, params=payload)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    main()