# !/usr/env/python


def filecmp(path_1, path_2):
    l1 = l2 = " "
    with open(path_1) as f1, open(path_2) as f2:
        while l1 != "" and l2 != "":
            l1 = f1.readline().strip()
            l2 = f2.readline().strip()
            if l1 != l2:
                return False
    return True
