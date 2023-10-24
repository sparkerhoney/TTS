import os

def list_files(directory):
    """특정 디렉토리에 있는 파일 이름들을 반환합니다."""
    with os.scandir(directory) as entries:
        return [entry.name for entry in entries if entry.is_file()]

# 실행 예제
if __name__ == '__main__':
    directory_path = './'  # 현재 디렉토리를 기본으로 설정. 필요에 따라 변경 가능.
    files = list_files(directory_path)
    for file in files:
        print(file)
