from eurosat import data

def main():
    print("Hello from eurosat!")
    path = data.download_data()
    rgb_data = data.RGBData(path)

    dataset, label_map = rgb_data.load_data()


if __name__ == "__main__":
    main()
