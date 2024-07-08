import os

def print_directory_structure(root_dir, indent=""):
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            print(f"{indent}{item}/")
            print_directory_structure(item_path, indent + "    ")
        else:
            print(f"{indent}{item}")

if __name__ == "__main__":
    root_directory = "."  # Change to the path of the directory you want to start from
    print(f"Directory structure of {os.path.abspath(root_directory)}:")
    print_directory_structure(root_directory)
