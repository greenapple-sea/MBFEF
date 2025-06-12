import os
import glob


def count_defects_in_dataset(dataset_root, subset):
    """
    计算指定数据集子集中的缺陷数量
    :param dataset_root: 数据集根路径
    :param subset: 子集名称 ('train', 'val' 或 'test')
    :return: 各类缺陷数量的字典
    """
    # 构建labels路径（与images同级目录）
    labels_path = os.path.join(dataset_root, 'labels', subset)

    if not os.path.exists(labels_path):
        return None

    label_files = glob.glob(os.path.join(labels_path, '*.txt'))

    defect_counts = {
        0: 0,  # missing_hole
        1: 0,  # mouse_bite
        2: 0,  # open_circuit
        3: 0,  # short
        4: 0,  # spur
        5: 0  # spurious_copper
    }

    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if parts:  # 确保行不为空
                    label = int(parts[0])
                    if label in defect_counts:
                        defect_counts[label] += 1

    return defect_counts


def main():
    # 数据集根路径
    dataset_root = '/root/autodl-tmp/pcb_2_001/pcb/PCB_Error_Detect/PCB_DATASET/Dataset_B/new_coco_form'

    # 要统计的子集
    subsets = ['train', 'val', 'test']

    # 计算并打印各子集的缺陷数量
    for subset in subsets:
        counts = count_defects_in_dataset(dataset_root, subset)
        if counts is not None:
            print(f"{subset} set defect counts:")
            print(f"  missing_hole (0): {counts[0]}")
            print(f"  mouse_bite (1): {counts[1]}")
            print(f"  open_circuit (2): {counts[2]}")
            print(f"  short (3): {counts[3]}")
            print(f"  spur (4): {counts[4]}")
            print(f"  spurious_copper (5): {counts[5]}")
            print(f"  Total defects: {sum(counts.values())}\n")
        else:
            print(f"{subset} set labels not found at: {os.path.join(dataset_root, 'labels', subset)}")


if __name__ == '__main__':
    main()
