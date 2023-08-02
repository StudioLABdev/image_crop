import cv2
from matplotlib import pyplot as plt
import numpy as np


def crop_image():
    image = cv2.imread("../musinsaCrawler/images/92.jpg", cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 임계값 설정 및 이진화
    _, binary_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

    # 이진화 이미지 출력
    # plt.imshow(binary_image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # 각 행에 있는 흰색 픽셀의 개수 계산
    white_pixel_counts = np.sum(binary_image == 255, axis=1)

    # 임계값을 설정하여 흰색 선이 있는 위치 찾기 (현재: 이미지 너비의 90% 이상이 흰색인 행)
    threshold = 0.9 * binary_image.shape[1]
    line_positions = np.where(white_pixel_counts > threshold)[0]

    # 연속된 흰색 선의 위치를 그룹화
    grouped_lines = []
    current_group = []

    for i in range(len(line_positions)):
        # 현재 위치와 이전 위치의 차이가 1보다 크면 새로운 그룹을 시작
        if i == 0 or (line_positions[i] - line_positions[i - 1] > 1):
            if current_group:
                grouped_lines.append(current_group)
            current_group = [line_positions[i]]
        else:
            current_group.append(line_positions[i])

    if current_group:
        grouped_lines.append(current_group)

    # 각 그룹의 중간 위치를 기준으로 이미지를 자를 위치 결정
    cut_positions = [(group[0], group[-1]) for group in grouped_lines]

    # 이미지를 수평으로 자르는 함수
    def cut_image(image, cut_positions):
        cut_images = []

        # 첫 번째 조각
        cut_images.append(image[0:cut_positions[0][0]])

        # 중간 조각들
        for i in range(len(cut_positions) - 1):
            cut_images.append(image[cut_positions[i][1] + 1:cut_positions[i + 1][0]])

        # 마지막 조각
        cut_images.append(image[cut_positions[-1][1] + 1:])

        return cut_images

    # 이미지 자르기
    cut_images = cut_image(image_rgb, cut_positions)

    # 너무 작은 조각 제외
    filtered_cut_images = [img for img in cut_images if img.shape[0] > 5]

    file_paths = []

    # cwd = os.getcwd()
    # print(cwd)
    for idx, img in enumerate(filtered_cut_images, 1):
        file_path = f"./cut_image_{idx}.jpg"
        cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        file_paths.append(file_path)
    print(file_paths)image = cv2.imread("../musinsaCrawler/images/92.jpg", cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 임계값 설정 및 이진화
    _, binary_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

    # 이진화 이미지 출력
    # plt.imshow(binary_image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # 각 행에 있는 흰색 픽셀의 개수 계산
    white_pixel_counts = np.sum(binary_image == 255, axis=1)

    # 임계값을 설정하여 흰색 선이 있는 위치 찾기 (현재: 이미지 너비의 90% 이상이 흰색인 행)
    threshold = 0.9 * binary_image.shape[1]
    line_positions = np.where(white_pixel_counts > threshold)[0]

    # 연속된 흰색 선의 위치를 그룹화
    grouped_lines = []
    current_group = []

    for i in range(len(line_positions)):
        # 현재 위치와 이전 위치의 차이가 1보다 크면 새로운 그룹을 시작
        if i == 0 or (line_positions[i] - line_positions[i - 1] > 1):
            if current_group:
                grouped_lines.append(current_group)
            current_group = [line_positions[i]]
        else:
            current_group.append(line_positions[i])

    if current_group:
        grouped_lines.append(current_group)

    # 각 그룹의 중간 위치를 기준으로 이미지를 자를 위치 결정
    cut_positions = [(group[0], group[-1]) for group in grouped_lines]

    # 이미지를 수평으로 자르는 함수
    def cut_image(image, cut_positions):
        cut_images = []

        # 첫 번째 조각
        cut_images.append(image[0:cut_positions[0][0]])

        # 중간 조각들
        for i in range(len(cut_positions) - 1):
            cut_images.append(image[cut_positions[i][1] + 1:cut_positions[i + 1][0]])

        # 마지막 조각
        cut_images.append(image[cut_positions[-1][1] + 1:])

        return cut_images

    # 이미지 자르기
    cut_images = cut_image(image_rgb, cut_positions)

    # 너무 작은 조각 제외
    filtered_cut_images = [img for img in cut_images if img.shape[0] > 5]

    file_paths = []

    # cwd = os.getcwd()
    # print(cwd)
    for idx, img in enumerate(filtered_cut_images, 1):
        file_path = f"./cut_image_{idx}.jpg"
        cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        file_paths.append(file_path)
    print(file_paths)


if __name__ == '__main__':
    crop_image()