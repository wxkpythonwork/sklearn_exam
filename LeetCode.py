#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.5
@author: wxk
@license: Apache Licence 
@file: LeetCode.py
@time: 2018/3/16 20:03
"""
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i in range(len(nums)-1):
            for j in range(i,len(nums)):
                if nums[i] + nums[j] == target:
                    a = [i,j]
                    return a

class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        row = len(array)-1
        col = len(array[0])-1
        j = 0
        while row>=0 and j<=col
            for j in range(col):
                if target > array[row][j]:
                    j += 1
                elif target < array[row][j]:
                    row -= 1
                else:
                    return True
            return False