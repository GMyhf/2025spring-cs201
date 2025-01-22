# $Typora$基本用法
Updated CST 2024 Nov 21 15:19

下载后用Typora查看效果更佳

[TOC]

源代码模式:com /

------

------

## 标题级别

设定:com 1~6

升降:com +/-

正文:com 0

------

------

## 文本处理

加粗:com B(Bold)或\**text**

斜体:com I(Italic)或\*text*

下划线:com U(Underline)或\<u>text\</u>

高亮:com shift H(Highlight)或\==text==

上标:\^text^

下标:\~text~

目录:[TOC]

分割线:opt com -或- - - Return

块引用:>text

------

------

## 列表

1. 无序列表:-/+/* space或opt com U(Unordered)
   增加/减少缩进:Tab或com ]/com [
   列表内换行:control Return
2. 有序列表:1. space或opt com O(Ordered)
   增加/减少缩进:Tab或com ]/com [
3. 任务列表:- space × 3 [x]或opt com X

------

------

## 各类引用

1. 行内代码:\`code\`或^`
   代码块:``` Return或opt com C(Code)
   (已经默认为python,更改参考CSDN教程)
2. 表格:opt com T(Table)
3. 引用:opt com Q(Quote)
4. 超链接(网页):com K或opt com L(Link)或\[title](URL)
   超链接(本文):com K或\[text](# name)
5. 脚注:opt com R或text\[^tag] 中间blabla结尾 \[^tag]:text

------

------

## 插入图片

1. 本地图片调用(不可更改路径)
   ①拷贝粘贴
   ②control com I输入路径
2. 网络图片调用
   control com I输入来源
3. iPic图床
   shift com U

------

------

## 图片排版

```html
<center> Return
#创建HTML,居中效果,左右同理

<img src = "来源"

	width = “缩放比例%" >
	#插入图片

\<br>
#不加则会并排

此处可加图注
```

若要在图注中加入公式则需图片形式插入
