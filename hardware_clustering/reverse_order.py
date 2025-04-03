def merge_count_split_inv(arr, temp_arr, left, right):
    """
    用于在归并排序中统计逆序对。
    :param arr: 原始数组
    :param temp_arr: 临时数组，用于存储排序后的结果
    :param left: 当前子数组的左边界
    :param right: 当前子数组的右边界
    :return: 逆序对的数量
    """
    if left == right:
        return 0

    mid = (left + right) // 2
    inv_count = 0

    # 递归分解数组并计算逆序对
    inv_count += merge_count_split_inv(arr, temp_arr, left, mid)
    inv_count += merge_count_split_inv(arr, temp_arr, mid + 1, right)

    # 合并两个子数组并计算逆序对
    inv_count += merge_and_count(arr, temp_arr, left, mid, right)

    return inv_count

def merge_and_count(arr, temp_arr, left, mid, right):
    """
    合并两个排序的子数组，并计算逆序对。
    :param arr: 原始数组
    :param temp_arr: 临时数组，用于存储合并后的数组
    :param left: 当前子数组的左边界
    :param mid: 当前子数组的中点
    :param right: 当前子数组的右边界
    :return: 逆序对的数量
    """
    i = left    # 左子数组的起始位置
    j = mid + 1 # 右子数组的起始位置
    k = left    # 合并后数组的位置
    inv_count = 0

    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            i += 1
        else:
            temp_arr[k] = arr[j]
            inv_count += (mid-i + 1)  # 逆序对的数量等于左子数组中剩余的元素数
            j += 1
        k += 1

    # 复制剩余的元素
    while i <= mid:
        temp_arr[k] = arr[i]
        i += 1
        k += 1

    while j <= right:
        temp_arr[k] = arr[j]
        j += 1
        k += 1

    # 将临时数组的内容复制回原数组
    for i in range(left, right + 1):
        arr[i] = temp_arr[i]

    return inv_count

def count_inversions(arr):
    """
    计算逆序对的总数
    :param arr: 输入数组
    :return: 逆序对的总数
    """
    n = len(arr)
    temp_arr = [0] * n
    return merge_count_split_inv(arr, temp_arr, 0, n-1)
def main():
    arr1 = [1,2,3,4]
    arr2 = [2,1,4,3]
    arr3 = [3,4,1,2]
    arr4 = [4,3,2,1]
    print(f"{arr1}逆序对的数量是：{count_inversions(arr1)}")
    print(f"{arr2}逆序对的数量是：{count_inversions(arr2)}")
    print(f"{arr3}逆序对的数量是：{count_inversions(arr3)}")
    print(f"{arr4}逆序对的数量是：{count_inversions(arr4)}")
if __name__ == "__main__":
    main()
