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
        freq = {}
        for x in nums:
            freq[x] = freq.get(x, 0) + 1
            print(freq)

        # 2) buat list pasangan (num, freq)
        items = list(freq.items())  # contoh: [(1,3), (2,2), (3,1)]

        # 3) bubble sort berdasarkan freq terbesar dulu (descending)
        n = len(items)
        for i in range(n):
            for j in range(0, n - i - 1):
                if items[j][1] < items[j + 1][1]:  # bandingkan frekuensi
                    items[j], items[j + 1] = items[j + 1], items[j]

        # 4) ambil k teratas
        return [num for num, _ in items[:k]]
    
    def encode(self, strs: List[str]) -> str:
        res = ""
        for s in strs:
            res += str(len(s)) + "#" + s
        return res

    def decode(self, s: str) -> List[str]:
        res = []
        i = 0

        while i < len(s):
            j = i
            while s[j] != "#":
                j += 1

            length = int(s[i:j])
            word = s[j + 1 : j + 1 + length]
            res.append(word)
            i = j + 1 + length

        return res

    def maxProfit(self, prices: List[int]) -> int:
        max_price = 0
        topProfit = 0

        for price in reversed(prices):
            max_price = max(max_price, price)
            topProfit = max(topProfit, max_price - price)
            print(max_price, topProfit) 

        return
    
    def maxProfit2(self, prices: List[int]) -> int:
        max_price = 0
        topProfit = 0

        for price in prices:
            max_price = max(max_price, price)
            topProfit = max(topProfit, max_price - price) 
            print(max_price, topProfit)

        return


code = LeetCode()


def encode(strs: List[str]) -> str:
    s = ""

    for i in strs:
        s += "#" + i

    return s

encoded = encode(["lintang", "dwi", "prasetyo"])
print(encoded)  # Output: "#lintang#dwi#prasetyo"