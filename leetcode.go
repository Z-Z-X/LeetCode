/*
 * @lc app=leetcode id=1 lang=golang
 *
 * [1] Two Sum
 */
package main

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

func main() {
	// var a=[][]int{{1,1},{0,0}}
	// fmt.Println(oddCells(2,2,a))
	// romanToInt("III")

	// var strs []string
	// fmt.Println(longestCommonPrefix(strs))

	// fmt.Println(isValid("){"))
	// fmt.Println(countAndSay(6))
	// lengthOfLastWord("a   ")

	// fmt.Println(addBinary("11", "1"))
	// fmt.Println(plusOne([]int{0}))
	// var a = [][]int{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	// rotate(a)
	// a := []int{1, 2, 2, 1}
	// b := []int{1}
	// fmt.Println(intersect(a, b))

	// strStr("mississippi", "issipi")

	// fmt.Println(isPalindrome2(".,"))
	// fmt.Println(myAtoi("+083472"))
	// fmt.Println(countPrimes(10))
	// var a = [][]int{{1, 4, 7, 11, 15}, {2, 5, 8, 12, 19}, {3, 6, 9, 16, 22}, {10, 13, 14, 17, 24}, {18, 21, 23, 26, 30}}
	// fmt.Println(searchMatrix(a, 5))
	// var a = []int{1, 3, -1, -3, 5, 3, 6, 7}
	// maxSlidingWindow(a, 3)

	numSquares(12)
}

// LeetCode 1
func twoSum(nums []int, target int) []int {
	for i, v := range nums {
		for index := i + 1; index < len(nums); index++ {
			if v+nums[index] == target {
				ret := []int{i, index}
				return ret
			}
		}
	}
	return nil
}

// LeetCode 7
func reverse(x int) int {
	b := x
	if x < 0 {
		b = -x
	}
	s := strconv.Itoa(b)
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	y, _ := strconv.Atoi(string(runes))

	if x < 0 {
		if -y < math.MinInt32 {
			return 0
		}
		return -y
	}
	if y > math.MaxInt32 {
		return 0
	}
	return y
}

type ListNode struct {
	Val  int
	Next *ListNode
}

// LeetCode 21
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	var r *ListNode
	if l1.Val >= l2.Val {
		r = l2
		r.Next = mergeTwoLists(l1, l2.Next)
	} else {
		r = l1
		r.Next = mergeTwoLists(l1.Next, l2)
	}
	return r
}

func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	var r *ListNode
	r = head
	for head.Next != nil {
		if head.Next.Val == head.Val {
			head.Next = head.Next.Next
		} else {
			head = head.Next
		}
	}
	return r
}

// LeetCode 203
func removeElements(head *ListNode, val int) *ListNode {
	if head == nil {
		return nil
	}
	var r *ListNode

	for head.Val == val {
		head = head.Next
		if head == nil {
			return nil
		}
	}
	if head.Next == nil {
		return head
	}
	r = head

	for t := head; t.Next != nil; {
		if t.Next.Val == val {
			t.Next = t.Next.Next
		} else {
			t = t.Next
		}
	}
	return r
}

// LeetCode 160
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	l1, l2 := headA, headB
	for l1 != l2 {
		if l1 == nil {
			l1 = headB
		} else {
			l1 = l1.Next
		}
		if l2 == nil {
			l2 = headA
		} else {
			l2 = l2.Next
		}
	}
	return l1
}

// LeetCode LCP1
func game(guess []int, answer []int) int {
	cnt := 0
	for i, v := range guess {
		if answer[i] == v {
			cnt++
		}
	}
	return cnt
}

// LeetCode 141
func hasCycle(head *ListNode) bool {
	if head == nil {
		return false
	}
	pA := head.Next
	if head.Next == nil {
		return false
	}
	pB := head.Next.Next
	for pA != nil && pB != nil {
		if pA.Val == pB.Val {
			return true
		}
		pA = pA.Next
		if pB.Next == nil {
			return false
		}
		pB = pB.Next.Next
	}
	return false
}

// LeetCode 206
func reverseList(head *ListNode) *ListNode {
	/* 递归方法
	if head == nil {
		return nil
	}
	res := reverseList(head.Next)
	if res == nil {
		return head
	}
	r:=res
	for res.Next != nil {
		res = res.Next
	}
	res.Next = head
	head.Next = nil

	return r
	*/

	/*迭代方法*/
	if head == nil {
		return nil
	}
	pre := head
	curr := head.Next
	pre.Next = nil
	for curr != nil {
		next := curr.Next
		curr.Next = pre
		pre = curr
		curr = next
	}
	return pre
}

// LeetCode 234
func isPalindrome(head *ListNode) bool {
	if head == nil {
		return true
	}

	fast, slow := head, head

	// 找到翻转中点
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}

	var pre *ListNode
	curr := slow
	for curr != nil {
		next := curr.Next
		curr.Next = pre
		pre = curr
		curr = next
	}

	for pre != nil {
		if pre.Val != head.Val {
			return false
		}
		pre = pre.Next
		head = head.Next
	}
	return true
}

// LeetCode 237
func deleteNode(node *ListNode) {
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}

// LeetCode 707
type MyLinkedList struct {
	Val  int
	Next *MyLinkedList
}

/** Initialize your data structure here. */
func Constructor() MyLinkedList {
	return MyLinkedList{Val: 0, Next: nil}
}

/** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
func (this *MyLinkedList) Get(index int) int {
	for i := 0; i < index; i++ {
		this = this.Next
	}
	return this.Val
}

/** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
func (this *MyLinkedList) AddAtHead(val int) {
	head := this
	newNode := &MyLinkedList{
		Val:  val,
		Next: head,
	}
	this = newNode
}

/** Append a node of value val to the last element of the linked list. */
func (this *MyLinkedList) AddAtTail(val int) {
	newNode := &MyLinkedList{
		Val:  val,
		Next: nil,
	}
	for this != nil {
		this = this.Next
	}
	this = newNode
}

/** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
func (this *MyLinkedList) AddAtIndex(index int, val int) {
	if index <= 0 {
		this.AddAtHead(val)
	}
	for i := 0; i < index-1; i++ {
		this = this.Next
	}
	newNode := &MyLinkedList{
		Val:  val,
		Next: this.Next,
	}
	this.Next = newNode
}

/** Delete the index-th node in the linked list, if the index is valid. */
func (this *MyLinkedList) DeleteAtIndex(index int) {
	for i := 0; i < index-1; i++ {
		this = this.Next
	}
	if this.Next != nil {
		this.Next = this.Next.Next
	}
}

func (this *MyLinkedList) Display() {
	for this != nil {
		fmt.Printf("%d-", this.Val)
		this = this.Next
	}
}

// LeetCode 876
func middleNode(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	fast, slow := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	return slow
}

// LeetCode 35
func searchInsert(nums []int, target int) int {
	if target < nums[0] {
		return 0
	}
	for i, v := range nums {
		if v == target {
			return i
		}
		if i == len(nums)-1 {
			return len(nums)
		}
		if nums[i] < target && nums[i+1] > target {
			return i + 1
		}
	}
	return 0
}

// LeetCode 35 二分法
func searchInsert2(nums []int, target int) int {
	if len(nums) == 0 {
		return 0
	}
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (left + right) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return left
}

// LeetCode 53
func maxSubArray(nums []int) int {
	// var max =nums[0]
	// for i := 0; i < len(nums); i++ {
	// 	var maxTemp int
	// 	for j := i; j < len(nums); j++ {
	// 		maxTemp += nums[j]
	// 		if maxTemp > max {
	// 			max = maxTemp
	// 		}
	// 	}
	// }
	// return max
	var max = nums[0]
	var maxTemp = 0
	for _, v := range nums {
		if maxTemp > 0 {
			maxTemp += v
		} else {
			maxTemp = v
		}
		if maxTemp > max {
			max = maxTemp
		}
	}
	return max
}

// LeetCode 88
func merge(nums1 []int, m int, nums2 []int, n int) {
	// 解法1
	// for i:=0;i<n;i++{
	// 	nums1[m+i]=nums2[i]
	// }
	// sort.Ints(nums1)

	// 解法2 双指针
	tempArray := make([]int, m)
	copy(tempArray, nums1[:m])
	p1, p2, p := 0, 0, 0
	for p1 < m && p2 < n {
		if tempArray[p1] > nums2[p2] {
			nums1[p] = nums2[p2]
			p++
			p2++
		} else {
			nums1[p] = tempArray[p1]
			p++
			p1++
		}
	}
	if p1 < m {
		copy(nums1[p1+n:], tempArray[p1:])
	}
	if p2 < n {
		copy(nums1[p2+m:], nums2[p2:])
	}
}

// LeetCode 977
func sortedSquares(A []int) []int {
	for i, v := range A {
		A[i] = v * v
	}
	sort.Ints(A)
	return A
}

// LeetCode 118
func generate(numRows int) [][]int {
	rows := make([][]int, numRows)
	for idx, _ := range rows {
		rows[idx] = make([]int, idx+1)
		for idx2 := 0; idx2 < idx+1; idx2++ {
			if idx2 == 0 || idx2 == idx {
				rows[idx][idx2] = 1
			} else {
				rows[idx][idx2] = rows[idx-1][idx2-1] + rows[idx-1][idx2]
			}
		}
	}
	return rows
}

// LeetCode 119
func getRow(rowIndex int) []int {
	var rows = make([]int, rowIndex+1)
	rows[0] = 1
	for i := 1; i < rowIndex+1; i++ {
		for j := i; j >= 1; j-- {
			rows[j] += rows[j-1]
		}
	}
	return rows
}

// LeetCode 867
func transpose(A [][]int) [][]int {
	m := len(A)
	n := len(A[0])
	res := make([][]int, n)
	for i := 0; i < n; i++ {
		res[i] = make([]int, m)
		for j := 0; j < m; j++ {
			res[i][j] = A[j][i]
		}
	}
	return res
}

// LeetCode 283
func moveZeroes(nums []int) {
	cnt := 0
	// 双指针，一个指针寻找非0元素，一个指针记录已有非0数组位置
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[cnt] = nums[i]
			cnt++
		}
	}
	for j := cnt; j < len(nums); j++ {
		nums[j] = 0
	}
	fmt.Println(nums)
}

// LeetCode 121 转化为正向两数最大差值
func maxProfit(prices []int) int {
	// if len(prices)==0{
	// 	return 0
	// }
	// min:=prices[0]
	// res:=0
	// for i:=0;i<len(prices)-1;i++{
	// 	max:=0
	// 	if prices[i] < min{
	// 		min=prices[i]
	// 	}
	// 	for j:=i+1;j<len(prices);j++{
	// 		if prices[j] > max{
	// 			max=prices[j]
	// 		}
	// 	}
	// 	if res < max-min{
	// 		res=max-min
	// 	}
	// }
	// return res

	if len(prices) == 0 {
		return 0
	}
	min := prices[0]
	maxRes := 0
	res := 0
	for i := 0; i < len(prices)-1; i++ {
		if prices[i] < min {
			min = prices[i]
		}
		if res = prices[i] - min; res > 0 && res > maxRes {
			maxRes = res
		}
	}
	return maxRes
}

// LeetCode 167
func twoSum2(numbers []int, target int) []int {
	// res := []int{0, 0}
	// first := numbers[0]
	// for i := 0; i < len(numbers)-1; i++ {
	// 	second := target - first
	// 	for j := i+1; j < len(numbers); j++ {
	// 		if numbers[j] == second {
	// 			res[0] = i + 1
	// 			res[1] = j + 1
	// 			return res
	// 		}
	// 	}
	// 	first = numbers[i+1]
	// }
	// return res

	left, right := 0, len(numbers)-1
	res := []int{0, 0}
	for left < right {
		if numbers[left]+numbers[right] == target {
			res[0] = left + 1
			res[1] = right + 1
			return res
		}
		if numbers[left]+numbers[right] < target {
			left++
		} else {
			right--
		}
	}
	return res
}

// LeetCode 169 TODO 投票法
func majorityElement(nums []int) int {
	sort.Ints(nums)
	return nums[len(nums)/2]
}

// LeetCode 219
// func containsNearbyDuplicate(nums []int, k int) bool {
//
// }

// LeetCode 268
func missingNumber(nums []int) int {
	var sum int
	var sum2 int
	for _, v := range nums {
		sum += v
	}
	sum2 = (len(nums) + 0) * (len(nums) + 1) / 2
	return sum2 - sum
}

// LeetCode 448
func findDisappearedNumbers(nums []int) []int {
	res := make([]int, 0)
	for _, v := range nums {
		nums[(v-1)%len(nums)] += len(nums)
	}
	for i, v := range nums {
		if v <= len(nums) {
			res = append(res, i+1)
		}
	}
	return res
}

// LeetCode 485
func findMaxConsecutiveOnes(nums []int) int {
	p, max := -1, 0
	for idx, val := range nums {
		if val != 0 && p == -1 {
			p = idx
		} else if val == 0 {
			if idx-p > max && p >= 0 {
				max = idx - p
			}
			p = -1
		}
	}
	if len(nums)-p > max && p >= 0 {
		max = len(nums) - p
	}
	return max
}

// LeetCode 509
func fib(N int) int {
	if N == 0 {
		return 0
	}
	if N == 1 {
		return 1
	}
	return fib(N-1) + fib(N-2)

	/* 非递归方法

	func fib(N int) int {
	    if N == 0 {
	        return 0
	    }
	    if N == 1 {
	        return 1
	    }

	    s := make([]int, N+1)
	    s[0] = 0
	    s[1] = 1

	    for i:=2; i <= N; i++ {
	        s[i] = s[i-2] + s[i-1]
	    }
	    return s[N]
	}

	*/
}

// LeetCode 605
func canPlaceFlowers(flowerbed []int, n int) bool {
	// 边界可种连续0长度l/2，中间可种连续0长度 (l-1) / 2,
	p := -1
	num := 0
	for idx, val := range flowerbed {
		if val != 1 && p == -1 {
			p = idx
		} else if val == 1 && p != -1 {
			if p == 0 {
				num += (idx - p) / 2
				p = -1
			} else {
				num += (idx - p - 1) / 2
				p = -1
			}
		}
	}
	if p != -1 {
		// 未出现1
		if p == 0 {
			num += (len(flowerbed) + 1) / 2
		} else { // 以0结尾
			num += (len(flowerbed) - p) / 2
		}
	}
	if n > num {
		return false
	}
	return true
}

// LeetCode 532 TODO 错误待修改
func findPairs(nums []int, k int) int {
	n := 0
	res := make([]int, 0)
	tempMap := map[int]byte{}
	for _, val := range nums {
		l := len(tempMap)
		tempMap[val] = 0
		if len(tempMap) != l {
			res = append(res, val)
		}
	}
	for idx, val := range res {
		for j := idx + 1; j < len(res); j++ {
			if (val-res[j])*(val-res[j]) == k*k {
				n++
			}
		}
	}
	return n
}

// LeetCode 561
func arrayPairSum(nums []int) int {
	sort.Ints(nums)
	res := 0
	for i := 0; i < len(nums); i += 2 {
		res += nums[i]
	}
	return res
}

// LeetCode 566
func matrixReshape(nums [][]int, r int, c int) [][]int {
	_r := len(nums)
	_c := len(nums[0])
	if _c*_r != c*r {
		return nums
	}
	var res [][]int
	for i := 0; i < r; i++ {
		var row []int
		for j := 0; j < c; j++ {
			temp := i*c + j
			row = append(row, nums[temp/_c][temp%_c])
		}
		res = append(res, row)
	}
	return res
}

// LeetCode 581 TODO 错误待修改
func findUnsortedSubarray(nums []int) int {

	min, max := nums[0], nums[len(nums)-1]
	for _, val := range nums {
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}
	p, q := 0, len(nums)-1
	for i := 0; i < len(nums)-1 && p < q; i++ {
		if nums[i] <= nums[i+1] {
			p++
		} else {
			break
		}
	}
	for j := len(nums) - 1; j > 0 && p < q; j-- {
		if nums[j] >= nums[j-1] {
			q--
		} else {
			break
		}
	}
	if min != nums[0] {
		p = 0
	}
	if max != nums[len(nums)-1] {
		q = len(nums) - 1
	}
	if min == max {
		return 0
	}
	if p == q {
		return 0
	}
	return q - p + 1
}

// LeetCode 628
// 首先想到排序，乘最大三个数，考虑两个最小负数相乘大于倒数2、3，
func maximumProduct(nums []int) int {
	sort.Ints(nums)
	t1 := nums[0] * nums[1]
	t2 := nums[len(nums)-2] * nums[len(nums)-3]
	if t1 > t2 {
		return t1 * nums[len(nums)-1]
	}
	return t2 * nums[len(nums)-1]
}

// LeetCode 661
func imageSmoother(M [][]int) [][]int {
	ans := make([][]int, len(M))
	for i := 0; i < len(M); i++ {
		ans[i] = make([]int, len(M[i]))
		for j := 0; j < len(M[i]); j++ {
			ans[i][j] = cal(M, i, j)
		}
	}
	return ans
}

var dirR = []int{-1, 1, 0, 0, -1, -1, 1, 1}
var dirC = []int{0, 0, -1, 1, -1, 1, -1, 1}

func cal(M [][]int, i, j int) int {
	count := 1
	sum := M[i][j]
	for k := 0; k < len(dirR); k++ {
		nextR := i + dirR[k]
		nextC := j + dirC[k]
		if nextR >= 0 && nextR < len(M) && nextC >= 0 && nextC < len(M[i]) {
			count++
			sum += M[nextR][nextC]
		}
	}
	return sum / count
}

// LeetCode 697
func findShortestSubArray(nums []int) int {
	// 　包含所有最高频元素的最短子数组

	max := 0
	var minSub = len(nums)
	left := 0
	right := 0
	var maxKey []int
	m := make(map[int]int)
	for _, val := range nums {
		m[val]++
	}
	for key, val := range m {
		if val > max {
			max = val
			maxKey = []int{key}
		} else if val == max {
			maxKey = append(maxKey, key)
		}
	}
	if max == 1 {
		return 1
	}
	for _, v := range maxKey {
		var findL, findR = false, false
		for i, j := 0, len(nums)-1; i < j && !(findL && findR); {
			if nums[i] != v {
				i++
			} else {
				findL = true
				left = i
			}
			if nums[j] != v {
				j--
			} else {
				findR = true
				right = j
			}
		}
		if right-left+1 < minSub {
			minSub = right - left + 1
		}
	}
	return minSub
}

// leetcode 766
func isToeplitzMatrix(matrix [][]int) bool {
	for i := 1; i < len(matrix); i++ {
		mat1 := matrix[i][1:]
		mat2 := matrix[i-1][:len(matrix[i-1])-1]
		// if !reflect.DeepEqual(mat1,mat2){
		// 	return false
		// }
		for i, v := range mat1 {
			if ! (mat2[i] == v) {
				return false
			}
		}
	}
	return true
}

// Leetcode 832
func flipAndInvertImage(A [][]int) [][]int {
	for idx := 0; idx < len(A); idx++ {
		for i, j := 0, len(A[0])-1; i <= j; {
			A[idx][i], A[idx][j] = A[idx][j]^1, A[idx][i]^1
			i++
			j--
		}
	}
	return A
}

// Leetcode 905
func sortArrayByParity(A []int) []int {
	flagOdd := len(A) - 1
	flagEven := 0
	var t = make([]int, len(A))
	for _, val := range A {
		if val%2 == 0 {
			t[flagEven] = val
			flagEven++
		} else {
			t[flagOdd] = val
			flagOdd--
		}
	}
	return t
}

// LeetCode 922
func sortArrayByParityII(A []int) []int {
	flagOdd := 1
	flagEven := 0
	var t = make([]int, len(A))
	for _, val := range A {
		if val%2 == 0 {
			t[flagEven] = val
			flagEven += 2
		} else {
			t[flagOdd] = val
			flagOdd += 2
		}
	}
	return t
}

// LeetCode 888
func fairCandySwap(A []int, B []int) []int {
	sumA := 0
	sumB := 0
	mapB := make(map[int]bool)
	for _, val := range A {
		sumA += val
	}
	for _, val := range B {
		sumB += val
		mapB[val] = true
	}
	avg := (sumA + sumB) / 2
	subA := avg - sumA
	for i := 0; i < len(A); i++ {
		if _, ok := mapB[A[i]+subA]; ok {
			return []int{A[i], A[i] + subA}
		}
	}
	return nil
}

// LeetCode 896
func isMonotonic(A []int) bool {
	flagComfirm := false
	flagInc := false
	for i := 0; i < len(A)-1; i++ {
		if A[i]-A[i+1] > 0 && !flagComfirm {
			flagComfirm = true
			flagInc = false
		} else if A[i]-A[i+1] < 0 && !flagComfirm {
			flagComfirm = true
			flagInc = true
		}
		if A[i]-A[i+1] > 0 && flagComfirm && flagInc {
			return false
		}
		if A[i]-A[i+1] < 0 && flagComfirm && !flagInc {
			return false
		}
	}
	return true
}

// LeetCode 985
func sumEvenAfterQueries(A []int, queries [][]int) []int {
	ans := make([]int, 0)
	sum := 0
	for _, val := range A {
		if val%2 == 0 {
			sum += val
		}
	}
	for i := 0; i < len(queries); i++ {
		idx := queries[i][1]
		if A[idx]%2 == 0 {
			sum -= A[idx]
		}
		A[idx] += queries[i][0]
		if A[idx]%2 == 0 {
			sum += A[idx]
		}
		ans = append(ans, sum)
	}
	return ans
}

// LeetCode 999 当前所在单元格四个方向上是否有pawn，两者之间是否有bishop
func numRookCaptures(board [][]byte) int {
	row := 0
	col := 0
	count := 0
	for idx, _ := range board {
		for idx2, _ := range board[idx] {
			if board[idx][idx2] == []byte("R")[0] {
				row = idx
				col = idx2
			}
		}
	}
	for i := row - 1; i > 0; i-- {
		if board[i][col] == []byte("B")[0] {
			break
		}
		if board[i][col] == []byte("p")[0] {
			count++
			break
		}
	}
	for i := row + 1; i < 8; i++ {
		if board[i][col] == []byte("B")[0] {
			break
		}
		if board[i][col] == []byte("p")[0] {
			count++
			break
		}
	}
	for i := col - 1; i > 0; i-- {
		if board[row][i] == []byte("B")[0] {
			break
		}
		if board[row][i] == []byte("p")[0] {
			count++
			break
		}
	}
	for i := col + 1; i < 8; i++ {
		if board[row][i] == []byte("B")[0] {
			break
		}
		if board[row][i] == []byte("p")[0] {
			count++
			break
		}
	}

	return count
}

// Leetcode 1002 题目可以抽象成求A中字符出现的交集
func commonChars(A []string) []string {
	res := make([]string, 0)
	m := make(map[string]int)
	for _, char := range A[0] {
		m[string(char)] += 1
	}
	for i := 1; i < len(A); i++ {
		_m := make(map[string]int)
		for _, char := range A[i] {
			_m[string(char)] += 1
		}
		for k := range m {
			if value, ok := _m[k]; !ok {
				delete(m, k)
			} else {
				if value < m[k] {
					m[k] = value
				}
			}
		}
	}
	for k, v := range m {
		for ; v > 0; v-- {
			res = append(res, k)
		}
	}
	return res
}

// Leetcode 1013
func canThreePartsEqualSum(A []int) bool {
	// solution1 2712ms 7.3MB
	// sum1, sum3 := A[0], 0
	// for i := 0; i < len(A)-2; i++ {
	// 	sum2 := A[i+1]
	// 	for p := i+2; p < len(A); p++ {
	// 		sum3 += A[p]
	// 	}
	// 	for j := i + 2; j < len(A); j++ {
	// 		if sum1 == sum2 && sum2 == sum3 {
	// 			return true
	// 		}
	// 		sum2 += A[j]
	// 		sum3 -= A[j]
	// 	}
	// 	sum1 += A[i+1]
	// }
	// return false
	sum, avg := 0, 0
	sum1 := 0
	left, right := len(A), 0
	for _, v := range A {
		sum += v
	}
	avg = sum / 3
	if sum%3 != 0 {
		return false
	}
	for i := 0; i < len(A)-2; i++ {
		sum1 += A[i]
		if avg == sum1 {
			left = i
			sum1 = 0
			break
		}
	}
	if left == len(A) {
		return false
	}
	for i := left + 1; i < len(A); i++ {
		sum1 += A[i]
		if avg == sum1 {
			right = i
			break
		}
	}
	if left < right {
		return true
	}
	return false
}

// Leetcode 1089
func duplicateZeros(arr []int) {
	for i := 0; i < len(arr)-1; i++ {
		if arr[i] == 0 {
			for j := len(arr) - 1; j > i+1; j-- {
				arr[j] = arr[j-1]
			}
			arr[i+1] = 0
			i++
		}
	}
}

// Leetcode 1252
func oddCells(n int, m int, indices [][]int) int {
	var rows = make([]int, n)
	var cols = make([]int, m)
	var res = 0
	for idx, _ := range indices {
		rows[indices[idx][0]]++
		cols[indices[idx][1]]++
	}
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			if (rows[i]+cols[j])%2 > 0 {
				res++
			}
		}
	}
	return res
}

// Leetcode 9
func isPalindrome1(x int) bool {
	if x < 0 {
		return false
	}
	s := strconv.Itoa(x)
	for i, j := 0, len(s); i < j; {
		if s[i] != s[j] {
			return false
		}
		i++
		j--
	}
	return true
}

// LeetCode 13
// I             1
// V             5
// X             10
// L             50
// C             100
// D             500
// M             1000
func romanToInt(s string) int {
	var sum = 0
	for i := len(s) - 1; i >= 0; i-- {
		switch string(s[i]) {
		case "I":
			sum += 1
		case "V":
			if i-1 >= 0 && string(s[i-1]) == "I" {
				sum += 4
				i--
			} else {
				sum += 5
			}
		case "X":
			if i-1 >= 0 && string(s[i-1]) == "I" {
				sum += 9
				i--
			} else {
				sum += 10
			}
		case "L":
			if i-1 >= 0 && string(s[i-1]) == "X" {
				sum += 40
				i--
			} else {
				sum += 50
			}
		case "C":
			if i-1 >= 0 && string(s[i-1]) == "X" {
				sum += 90
				i--
			} else {
				sum += 100
			}
		case "D":
			if i-1 >= 0 && string(s[i-1]) == "C" {
				sum += 400
				i--
			} else {
				sum += 500
			}
		case "M":
			if i-1 >= 0 && string(s[i-1]) == "C" {
				sum += 900
				i--
			} else {
				sum += 1000
			}
		}
	}
	return sum
}

// leetcode 14
func longestCommonPrefix(strs []string) string {
	if len(strs) == 0 {
		return ""
	}
	var maxStr = strs[0]
	for i := 1; i < len(strs); i++ {
		for idx, val := range maxStr {
			if idx < len(strs[i]) {
				if val != rune(strs[i][idx]) {
					maxStr = maxStr[:idx]
					break
				}
			} else {
				maxStr = strs[i]
				break
			}
		}
	}
	return maxStr
}

// leetcode 20 待解决
func isValid(s string) bool {
	if len(s) == 0 {
		return true
	}
	if len(s)%2 != 0 {
		return false
	}
	var stack []string
	for _, val := range s {
		stack = append(stack, string(val))
		switch stack[len(stack)-1] {
		case "}":
			if len(stack) < 2 {
				return false
			}
			if stack[len(stack)-2] == "{" {
				stack = stack[:len(stack)-2]
			}
		case ")":
			if len(stack) < 2 {
				return false
			}
			if stack[len(stack)-2] == "(" {
				stack = stack[:len(stack)-2]
			}
		case "]":
			if len(stack) < 2 {
				return false
			}
			if stack[len(stack)-2] == "[" {
				stack = stack[:len(stack)-2]
			}
		}
	}
	if len(stack) != 0 {
		return false
	}
	return true
}

// leetcode 38
func countAndSay(n int) string {
	if n == 1 {
		return "1"
	}
	if n == 2 {
		return "11"
	}
	var res string
	str := countAndSay(n - 1)
	var cnt = 1
	var number = str[0]
	for index := 1; index < len(str); index++ {
		if str[index] == number {
			cnt++
		} else {
			res += strconv.Itoa(cnt)
			res += string(number)
			number = str[index]
			cnt = 1
		}
	}
	res += strconv.Itoa(cnt)
	res += string(number)
	return res
}

// leetcode 58
func lengthOfLastWord(s string) int {
	strs := strings.Split(s, " ")
	if len(strs) == 0 {
		return 0
	}
	for i := len(strs) - 1; i >= 0; i-- {
		if len(strs[i]) != 0 {
			return len(strs[i])
		}
	}
	return 0
}

// leetcode 67
func addBinary(a string, b string) string {
	var maxLen, subLen int
	var carry = byte(0)
	var res string
	lenA := len(a)
	lenB := len(b)
	if lenA > lenB {
		maxLen = lenA
		subLen = lenA - lenB
		b = strings.Repeat("0", subLen) + b
		// for i := 0; i < subLen; i++ {
		// 	b = "0" + b
		// }
	} else {
		maxLen = lenB
		subLen := lenB - lenA
		a = strings.Repeat("0", subLen) + a
		// for i := 0; i < subLen; i++ {
		// 	a = "0" + a
		// }
	}

	for i := maxLen - 1; i >= 0; i-- {
		r := (a[i] - 48 + b[i] - 48 + carry) % 2
		carry = (a[i] - 48 + b[i] - 48 + carry) / 2
		res = strconv.Itoa(int(r)) + res
	}
	if carry != uint8(0) {
		res = strconv.Itoa(int(carry)) + res
	}
	return res
}

// leetcode 771
func numJewelsInStones(J string, S string) int {
	var sum int
	var m = make(map[byte]int)
	for i, _ := range J {
		m[J[i]] = 0
	}
	for i, _ := range S {
		if _, ok := m[S[i]]; ok {
			m[S[i]]++
		}
	}
	for _, v := range m {
		sum += v
	}
	return sum
}

func intersect(nums1 []int, nums2 []int) []int {
	// 解法1 双循环遍历
	/*
		var res []int
		var m =make([]bool,len(nums2))
		for i:=0;i<len(nums1) ;i++  {
			for j:=0;j<len(nums2);j++{
				if nums1[i] ==nums2[j] && !m[j]{
					res = append(res, nums2[j])
					m[j]=true
					break
				}
			}
		}
		return res
	*/
	// 解法2 map存储频数
	var res []int
	var m = make(map[int]int)
	for _, i2 := range nums1 {
		m[i2]++
	}
	for _, i2 := range nums2 {
		if _, ok := m[i2]; ok && m[i2] > 0 {
			res = append(res, i2)
			m[i2]--
		}
	}
	return res
}

// 加1
func plusOne(digits []int) []int {
	var inc = 1
	for i := len(digits) - 1; i >= 0; i-- {
		remain := (digits[i] + inc) % 10
		if remain < digits[i] {
			inc = 1
		} else {
			inc = 0
		}
		digits[i] = remain
	}
	if inc != 0 {
		var res = make([]int, 0)
		res = append(res, 1)
		res = append(res, digits...)
		return res
	}
	return digits
}

// 有效的数独
func isValidSudoku(board [][]byte) bool {
	var row = make([][]bool, 9)
	var col = make([][]bool, 9)
	var block = make([][]bool, 9)
	for i := 0; i < 9; i++ {
		row[i] = make([]bool, 9)
		col[i] = make([]bool, 9)
		block[i] = make([]bool, 9)
	}
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if board[i][j] != byte('.') {
				number := board[i][j] - byte('1')
				blockIndex := i/3*3 + j/3
				if row[i][number] || col[j][number] || block[blockIndex][number] {
					return false
				}
				row[i][number] = true
				col[j][number] = true
				block[blockIndex][number] = true
			}
		}
	}
	return true
}

// leetcode 48 旋转图像
func rotate(matrix [][]int) {
	for i := 0; i < len(matrix); i++ {
		for j := i; j < len(matrix); j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
	for i := 0; i < len(matrix); i++ {
		p := len(matrix) - 1
		for j := 0; j < len(matrix)/2; j++ {
			matrix[i][j], matrix[i][p] = matrix[i][p], matrix[i][j]
			p--
		}
	}
	fmt.Println(matrix)
}

// leetcode 387 字符串中的第一个唯一字符
func firstUniqChar(s string) int {
	var m = make(map[int32]int)
	for _, val := range s {
		m[val]++
	}
	for i, val := range s {
		if m[val] == 1 {
			return i
		}
	}
	return -1
}

// leetcode 242 有效的字母异位词
// 对两个字符串内容排序，比较，不相等就false
// 使用map,将s的字符依次放入累计数量，t的字符减数量,不存在 false, 计数不为0false
// 使用数组，而不是map
func isAnagram(s string, t string) bool {
	/*
		var m = make(map[rune]int)
		for _,v:=range s{
			m[v]++
		}
		for _,v:=range t {
			if _,ok:=m[v];ok{
				m[v]--
			}else{
				return false
			}
		}
		for _,v:=range m{
			if v!=0{return false}
		}
		return true
	*/
	var m = make([]int, 26)
	for _, v := range s {
		m[v-'a']++
	}
	for _, v := range t {
		m[v-'a']--
	}
	for _, v := range m {
		if v != 0 {
			return false
		}
	}
	return true
}

func strStr(haystack string, needle string) int {

	if len(needle) == 0 {
		return 0
	}
	if len(haystack) < len(needle) {
		return -1
	}
	for i := 0; i < len(haystack)-len(needle)+1; i++ {
		if haystack[i] == needle[0] {
			if haystack[i:i+len(needle)] == needle {
				return i
			}
		}
	}
	return -1

}

// leetcode 125 验证回文字符串
// 考虑头尾双指针
func isPalindrome2(s string) bool {
	/*
		var ch1 ,ch2 byte
		ch1=ch1&0xDF
		ch2=ch2&0x20
	*/
	r := []rune(s)
	var head, tail int
	head, tail = 0, len(s)-1
	for head < tail {
		if !unicode.IsLetter(r[head]) && !unicode.IsDigit(r[head]) {
			head++
			continue
		}
		if !unicode.IsLetter(r[tail]) && !unicode.IsDigit(r[tail]) {
			tail--
			continue
		}
		if unicode.ToLower(r[head]) == unicode.ToLower(r[tail]) {
			head++
			tail--
		} else {
			return false
		}
	}
	return true
}

// leetcode 字符串转换整数 (atoi)
func myAtoi(str string) int {
	var p int
	str = strings.TrimSpace(str)
	if len(str) < 1 || str[0] != '-' && str[0] != '+' && (str[0] < '0' || str[0] > '9') {
		return 0
	}
	if str[0] == '-' {
		if len(str) < 2 {
			return 0
		}
		if str[1] < '0' || str[1] > '9' {
			return 0
		}
	}
	for i := 1; i < len(str); i++ {
		if str[i] >= '0' && str[i] <= '9' {
			p = i
		} else {
			p = i - 1
			break
		}
	}
	str = str[:p+1]
	n, _ := strconv.Atoi(str)
	if n > math.MaxInt32 {
		return math.MaxInt32
	}
	if n < math.MinInt32 {
		return math.MinInt32
	}
	return n
}

// leetcode142 环形链表II
func detectCycle(head *ListNode) *ListNode {
	var cycle bool
	if head == nil {
		return nil
	}
	pA := head.Next
	if head.Next == nil {
		return nil
	}
	pB := head.Next.Next
	for pA != nil && pB != nil {
		if pA.Val == pB.Val {
			cycle = true
			break
		}
		pA = pA.Next
		if pB.Next == nil {
			return nil
		}
		pB = pB.Next.Next
	}

	if cycle {
		p := head
		for p != nil && pA != nil {
			if p == pA {
				return p
			}
			p = p.Next
			pA = pA.Next
		}
	}

	return nil
}

// leetcode 19 删除链表的倒数第N个节点
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	var cnt = 0
	p := head
	for p != nil {
		p = p.Next
		cnt++
	}
	q := head
	if cnt-n == 0 {
		return head.Next
	}
	for i := 0; i < cnt-n-1; i++ {
		if q.Next != nil {
			q = q.Next
		}
	}
	if q.Next.Next != nil {
		q.Next = q.Next.Next
	} else {
		q.Next = nil
	}

	return head
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// leetcode 二叉树的最大深度
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	var leftDepth, rightDepth = 1, 1
	if root.Left != nil {
		leftDepth += maxDepth(root.Left)
	}
	if root.Right != nil {
		rightDepth += maxDepth(root.Right)
	}
	if rightDepth > leftDepth {
		return rightDepth
	}
	return leftDepth
}

func isValidBST(root *TreeNode) bool {
	return helper(root, math.MinInt64, math.MaxInt64)
}
func helper(root *TreeNode, lower, upper int) bool {
	var lValid, rValid = true, true
	if root == nil {
		return true
	}
	if lower >= root.Val || upper <= root.Val {
		return false
	}
	lValid = helper(root.Left, lower, root.Val)
	rValid = helper(root.Right, root.Val, upper)
	return lValid && rValid
}

// leetcode 101 对称二叉树
func isSymmetric(root *TreeNode) bool {
	return isMirror(root.Left, root.Right)
}
func isMirror(t1, t2 *TreeNode) bool {
	if t1 == nil && t2 == nil {
		return true
	}
	if t1 == nil || t2 == nil {
		return false
	}

	return (t1.Val == t2.Val) && isMirror(t1.Left, t2.Right) && isMirror(t1.Right, t2.Left)

}

// leetcode 102 二叉树层次遍历
func levelOrder(root *TreeNode) [][]int {

	var res [][]int
	var dfs func(root *TreeNode, level int)
	dfs = func(root *TreeNode, level int) {
		if root == nil {
			return
		}
		if len(res) == level {
			res = append(res, []int{})
		}
		res[level] = append(res[level], root.Val)

		dfs(root.Left, level+1)
		dfs(root.Right, level+1)
	}
	dfs(root, 0)
	return res
}

// leetcode 108 将有序数组转换为二叉搜索树
func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	mid := len(nums) / 2
	left := nums[:mid]
	right := nums[mid+1:]
	node := &TreeNode{
		Val:   nums[mid],
		Left:  sortedArrayToBST(left),
		Right: sortedArrayToBST(right),
	}
	return node
}

// leetcode 爬楼梯
func climbStairs(n int) int {
	if n <= 2 {
		return n
	}
	i1 := 1
	i2 := 2
	for i := 3; i <= n; i++ {
		temp := i1 + i2
		i1 = i2
		i2 = temp
	}
	return i2
}

// leetcode 198
func rob(nums []int) int {
	dp := make([]int, len(nums)+1)
	if len(nums) == 0 {
		return 0
	}
	dp[0] = 0
	dp[1] = nums[0]
	for i := 2; i <= len(nums); i++ {
		if dp[i-1] > dp[i-2]+nums[i-1] {
			dp[i] = dp[i-1]
		} else {
			dp[i] = dp[i-2] + nums[i-1]
		}
	}
	return dp[len(nums)]
}

// leetcode 412 fizz buzz
func fizzBuzz(n int) []string {
	var res []string
	for i := 1; i <= n; i++ {
		if i%3 == 0 && i%5 == 0 {
			res = append(res, "FizzBuzz")
			continue
		}
		if i%3 == 0 {
			res = append(res, "Fizz")
			continue
		}
		if i%5 == 0 {
			res = append(res, "Buzz")
			continue
		}
		res = append(res, strconv.Itoa(i))
	}
	return res
}

// leetcode 204 计数质数
func countPrimes(n int) int {
	/* 超时
	var cnt int
	var isPrime func(n int) bool
	isPrime= func(n int) bool {
		for i:=2;i*i<=n;i++{
			if n%i==0{
				return false
			}
		}
		return true
	}
	for i:=2;i<n;i++ {
		if isPrime(i){
			cnt++
		}
	}
	return cnt
	*/
	var count int
	var signs = make([]bool, n)
	for i := 2; i < n; i++ {
		if !signs[i] {
			count++
			for j := i * 2; j < n; j += i {
				signs[j] = true
			}
		}
	}
	return count
}

// leetcode
func isPowerOfThree(n int) bool {
	if n == 0 {
		return false
	}
	var a, b = 0, n
	for a == 0 {
		a = b % 3
		b = b / 3
	}
	if b == 0 && a == 1 {
		return true
	}
	return false
}

// leetcode
func hammingWeight(num uint32) int {
	s := strconv.FormatUint(uint64(num), 2)
	return strings.Count(s, "1")
}

// leetcode 461
func hammingDistance(x int, y int) int {
	t := x ^ y
	var ans int
	for t != 0 {
		t &= t - 1
		ans++
	}
	return ans
}

// leetcode
func reverseBits(num uint32) uint32 {
	var ret uint32
	ret = 0
	for i := 31; i >= 0; i-- {
		ret = ret | (((num >> uint(31-i)) & 1) << uint(i))
	}
	return ret
}

// leetcode 搜索二维矩阵II
func searchMatrix(matrix [][]int, target int) bool {
	if len(matrix) == 0 || len(matrix[0]) == 0 || target < matrix[0][0] || target > matrix[len(matrix)-1][len(matrix[0])-1] {
		return false
	}
	var maxRow, maxCol = len(matrix), len(matrix[0])
	for i := len(matrix) - 1; i >= 0; i-- {
		if matrix[i][0] < target {
			maxRow = i
			break
		} else if matrix[i][0] == target {
			return true
		}

	}
	for j := len(matrix[0]) - 1; j >= 0; j-- {
		if matrix[0][j] < target {
			maxCol = j
			break
		} else if matrix[0][j] == target {
			return true
		}
	}
	for i := 0; i <= maxRow; i++ {
		for j := 0; j <= maxCol; j++ {
			if matrix[i][j] == target {
				return true
			}
		}
	}
	return false
}

func maxProduct(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	minMulti := 1
	maxMulti := 1
	max := math.MinInt64
	for i := 0; i < len(nums); i++ {
		if nums[i] < 0 {
			temp := maxMulti
			maxMulti = minMulti
			minMulti = temp
		}
		if maxMulti*nums[i] > nums[i] {
			maxMulti = maxMulti * nums[i]
		} else {
			maxMulti = nums[i]
		}
		if minMulti*nums[i] < nums[i] {
			minMulti = minMulti * nums[i]
		} else {
			minMulti = nums[i]
		}
		if maxMulti > max {
			max = maxMulti
		}
	}
	return max
}

func productExceptSelf(nums []int) []int {
	var res = make([]int, len(nums))
	k := 1
	for i := 0; i < len(nums); i++ {
		res[i] = k
		k = k * nums[i]
	}
	k = 1
	for j := len(nums) - 1; j >= 0; j-- {
		res[j] *= k
		k = k * nums[j]
	}
	return res
}

// leetcode 334 递增的三元子序列
func increasingTriplet(nums []int) bool {
	if len(nums) < 3 {
		return false
	}
	var min = nums[0]
	var mid = math.MaxInt32
	for i := 1; i < len(nums); i++ {
		if nums[i] > min && nums[i] < mid {
			mid = nums[i]
			continue
		}
		if nums[i] > mid {
			return true
		}
		if nums[i] < min {
			min = nums[i]
			continue
		}
	}
	return false
}

// leetcode 数组中的第K个最大元素
func findKthLargest(nums []int, k int) int {
	sort.Ints(nums)
	return nums[len(nums)-k]
}

func maxSlidingWindow(nums []int, k int) []int {
	if len(nums) == 0 {
		return nil
	}
	var res []int
	var left = make([]int, len(nums))
	left[0] = nums[0]
	var right = make([]int, len(nums))
	right[len(nums)-1] = nums[len(nums)-1]
	for i := 1; i < len(nums); i++ {
		if i%k == 0 {
			left[i] = nums[i]
		} else {
			left[i] = IntMax(nums[i], left[i-1])
		}

		j := len(nums) - i - 1
		if (j+1)%k == 0 {
			right[j] = nums[j]
		} else {
			right[j] = IntMax(right[j+1], nums[j])
		}
	}
	for i := 0; i < len(nums)-k+1; i++ {
		res = append(res, IntMax(left[i-1+k], right[i]))
	}
	return res
}

func IntMax(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// leetcode 有序矩阵中第K小的元素
func kthSmallest(matrix [][]int, k int) int {
	if len(matrix) == 0 {
		return 0
	}
	var f func(n int) int
	f = func(n int) int {
		var count int
		var row, col = len(matrix) - 1, 0
		for row >= 0 && col < len(matrix) {
			if matrix[row][col] <= n {
				count += row + 1
				col++
			} else {
				row--
			}
		}
		return count
	}
	var left, right = matrix[0][0], matrix[len(matrix)-1][len(matrix)-1]
	for left < right {
		mid := left + (right-left)/2
		if f(mid) < k {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return left
}

// leetcode 完全平方数
func numSquares(n int) int {
	// 首先想到的递归方法，结果当然是超时了
	// 但是可以从这里推导出动态规划方法
	// 从1开始将每次最小次数保存，直到n,
	/*
		var min = n
		var tempRes = 0
		if n<=3{
			return n
		}
		for i := 1; i*i <= n; i++ {
			if i*i == n {
				return 1
			}
			tempRes = numSquares(n - i*i)+1
			if min > tempRes {
				min = tempRes
			}
		}
		return min
	*/

	// 动态规划
	if n <= 3 {
		return n
	}
	var dp = make([]int, n+1)
	dp[1] = 1
	dp[2] = 2
	dp[3] = 3
	for i := 4; i <= n; i++ {
		dp[i] = i
		for j := 1; i-j*j >= 0; j++ {
			dp[i] = IntMin(dp[i], dp[i-j*j]+1)
		}
	}
	return dp[n]
}

func IntMin(a, b int) int {
	if a > b {
		return b
	}
	return a
}

// leetcode 零钱兑换
func coinChange(coins []int, amount int) int {
	var dp = make([]int, amount+1)
	dp[0] = 0
	for i := 1; i <= amount; i++ {
		dp[i] = amount + 1
		for j := 0; j < len(coins); j++ {
			if i >= coins[j] {
				dp[i] = IntMin(dp[i], dp[i-coins[j]]+1)
			}
		}
	}
	// 未找到组合
	if dp[amount] > amount {
		return -1
	}
	return dp[amount]
}
