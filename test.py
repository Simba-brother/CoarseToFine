from collections import defaultdict
def test1():
    temp_list = [(0,2), (1,2),(2,2),(3,2),(4,2),(5,2)]
    b = [(idx,abs(v-3)) for idx,v in temp_list]
    
    print("temp_list:",temp_list)
    print("b:",b)
def test2():
    a_set = {4,4,3}
    a_set.add(2)
    print(a_set)
def test3():
    a_dict = defaultdict(set)
    a_dict[1] = [4,4,5]
    a_dict[2] = {9,9,8}
    print(a_dict)
def test4():
    data = [1,4,5,6]
    print(data.index(5))
def test5():
    a,_,_ = [1,2,3]
if __name__ == "__main__":
    test5()