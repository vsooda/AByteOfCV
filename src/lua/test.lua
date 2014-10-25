#!/usr/bin/lua

sum = 0
num = 1
while num <= 100 do
    sum = sum + num
    num = num + 1
end
print("sum = ", sum)


function fib(n)
    if n <  2 then return 1 end
    return fib(n-2) + fib(n-1)
end
