import requests

def main():
    # Replace with your actual Modal deployment URL
    url = "https://korona-lukasz--bnb-lora-inference-bnbloramodel-api-generate.modal.run"
    # url = "https://korona-lukasz--vllm-awq-lora-service-vllmloraserver-api--55782a.modal.run"
    # url = "https://korona-lukasz--vllm-compressor-inference-vllmcompressors-2fb78f.modal.run"

    # The data payload matching the dict expected by your endpoint
    data = {"prompt": "In the acute phase of HCV infection, what enzyme undergoes a rise and fall within a period of six months?"}

    # Send the POST request
    response = requests.post(url, json=data)

    # Print the JSON response from the server
    print(response.json())

if __name__ == "__main__":
    main()