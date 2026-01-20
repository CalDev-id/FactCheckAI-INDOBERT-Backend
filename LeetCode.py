from typing import List

class LeetCode:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dictOfList = {}

        for s in strs:
            setring = ''.join(sorted(s))
            if setring not in dictOfList:
                dictOfList[setring] = []
            dictOfList[setring].append(s)
        return list(dictOfList.values())
    
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i]+nums[j] == target:
                    return[i,j]

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {}
        topK = []

        for i in range(len(nums)):
            if nums[i] not in count:
                count[nums[i]] = 1
                for j in range(i+1, len(nums)):
                    if nums[i] == nums[j]:
                        count[nums[i]] = count[nums[i]] + 1
        for i, j in count.items():
            if j >= k:
                topK.append(i)
        return topK
    
    def bubbleSort(self, arr: List[int]) -> List[int]:
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] < arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
        
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {}

        for num in nums:
            count[num] = count.get(num, 0) + 1

        sorted_items = sorted(count.items(), key=lambda x: x[1], reverse=True)
        return [num for num, freq in sorted_items[:k]]

code = LeetCode()
# print(code.groupAnagrams(["eat","tea","tan","ate","nat","bat"]))
# task = code.twoSum([2,7,11,15], 9)
# print(task)

# data_karyawan = {'nama': 'Budi', 'usia': 30, 'kota': 'Jakarta'}
# semua_nilai = data_karyawan.values()
# print(data_karyawan)

print(code.bubbleSort([64, 11, 34, 25, 12, 22, 11, 90]))
