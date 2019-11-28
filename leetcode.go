/*
 * @lc app=leetcode id=1 lang=golang
 *
 * [1] Two Sum
 */
package main

import (
	"fmt"
	"sort"
	"strconv"
)

func main() {
	var a=[][]int{{1,1},{0,0}}
	fmt.Println(oddCells(2,2,a))
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
		return -y
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
	sum,avg:=0,0
	sum1:=0
	left,right:=len(A),0
	for _,v:=range A{
		sum+=v
	}
	avg=sum/3
	if sum%3!=0{
		return false
	}
	for i:=0;i<len(A)-2;i++{
		sum1+=A[i]
		if avg==sum1{
			left=i
			sum1=0
			break
		}
	}
	if left==len(A){
		return false
	}
	for i:=left+1;i<len(A);i++{
		sum1+=A[i]
		if avg==sum1{
			right=i
			break
		}
	}
	if left<right {
		return true
	}
	return false
}
// Leetcode 1089
func duplicateZeros(arr []int)  {
	for i:=0;i<len(arr)-1;i++{
		if arr[i]==0{
			for j:=len(arr)-1;j>i+1;j--{
				arr[j]=arr[j-1]
			}
			arr[i+1]=0
			i++
		}
	}
}

// Leetcode 1252
func oddCells(n int, m int, indices [][]int) int {
	var rows=make([]int, n)
	var cols=make([]int,m)
	var res=0
	for idx, _ := range indices {
		rows[indices[idx][0]]++
		cols[indices[idx][1]]++
	}
	for i:=0;i<n;i++{
		for j:=0;j<m;j++{
			if (rows[i]+cols[j])%2>0{
				res++
			}
		}
	}
	return res
}

// Leetcode 9
func isPalindrome1(x int) bool {
	if x < 0{
		return false
	}
	s:=strconv.Itoa(x)
	for i,j:=0,len(s);i<j;{
		if s[i]!=s[j]{
			return false
		}
		i++
		j--
	}
	return true
}