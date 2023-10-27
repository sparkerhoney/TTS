import os

def list_files(directory):
    with os.scandir(directory) as entries:
        return [entry.name for entry in entries if entry.is_file()]

# 실행 예제
if __name__ == '__main__':
    directory_path = 'SpeechT5/microsoft/speecht5_hifigan' 
    files = list_files(directory_path)
    for file in files:
        print(file)
