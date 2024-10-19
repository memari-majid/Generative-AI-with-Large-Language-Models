import os

def count_words_in_file(file_path):
    """Count the number of words in a given file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        words = content.split()
        return len(words)

def find_python_files_and_count_words(root_dir):
    """Recursively find all Python files and count the words in each."""
    total_word_count = 0
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(subdir, file)
                word_count = count_words_in_file(file_path)
                print(f"{file_path}: {word_count} words")
                total_word_count += word_count
    print(f"\nTotal words in all Python files: {total_word_count}")

if __name__ == "__main__":
    root_directory = "."  # Change this if you want to specify a different directory
    find_python_files_and_count_words(root_directory)
