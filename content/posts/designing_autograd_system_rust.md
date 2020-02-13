---
title: "Designing an Autograd System with Rust"
date: 2020-02-12T23:37:58-03:00
draft: true
---

# Why do it?

Well, there is certainly a gap in the ecosystem currently and at least
[some](https://github.com/rust-ml/discussion/issues/1) people 
are interested.

In theory, Rust can do whatever C/C++ does given enough effort. 
Since most of Pytorch/Tensorflow is C++/CUDA, at least the C++ part
 should be doable.

Also, I'm too naive to not try it, but as alluded 
in [a previous post](/blog/posts/current_deep_learning_ecosystem_from_a_rust_developer_perspective/)
I'm aware that it might not be so easy to follow Rust's ownership and borrowing
rules while providing similar ergonomics as Pytorch. 

# How it's done in Pytorch


Let's look at some minimal Pytorch code and try to figure out what is going on:

```Python
import torch;

# create two Tensors
x = torch.tensor([2], requires_grad=True, dtype=torch.float)
y = torch.tensor([3], requires_grad=True, dtype=torch.float)

# loop twice 
for k in range(2):
  z = (x*y)
  z.backward() # fills x and y gradients, so z must have a mutable reference of some kind to x and y
  print('x_grad =', x.grad)
  x.data = x.data-0.1*x.grad # here we are modifying x in-place, while z holds the mutable reference
  x.grad.data.zero_()  # zeroing the gradients of x
  print('x =', x)

print('z =', z)
```
Which runs just fine and outputs:
```Python 
x_grad = tensor([3.])
x = tensor([1.7000], requires_grad=True)
x_grad = tensor([3.])
x = tensor([1.4000], requires_grad=True)
z = tensor([5.1000], grad_fn=<MulBackward0>)
```

The code above this quite straight forward, but highlights why just translating this to Rust cannot work.

### Why this doesn't (and shouldn't) translate well to Rust

Python/Pytorch doesn't mind having multiple mutable references active at the same time.

This is *partially* mitigated due to Python ensuring memory safety by using reference counting (and also GC to break cycles),
 so it can, for the most part, avoid danging pointers. Not having real parallelism due to the
  [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) helps alleviate the classic data race/race conditions problems
   of mutable aliasing. 

Unfortunately, as a quote from this fantastic post about 
[The Problem With Single-threaded Shared Mutability](https://manishearth.github.io/blog/2015/05/17/the-problem-with-shared-mutability/)
points out:
                 
> My intuition is that code far away from my code might as well be in another thread, for all I can reason about what it will do to shared mutable state.
                  
   Having the underlying value or even type of a given variable being able to change from multiple places and even without  
  the variable being used at all can be jarring. Multiple mutable references also lead to other problems, such 
    as iterator invalidations.
  
#### Iterator Invalidation
```Python
lst = [1, 1, 2, 3]
for item in lst:
  if item == 1:
    lst.remove(item)

print(lst) # Prints: [1, 2, 3]
``` 
    
#### Underlying value changing indirectly
```Python
import torch;

def very_complex_long_fn(val):
    #Lots of code
    val.data = torch.tensor([20], requires_grad=True, dtype=torch.float)
    #Lots of code

x = torch.tensor([2], requires_grad=True, dtype=torch.float)
y = torch.tensor([3], requires_grad=True, dtype=torch.float)

a = [x, y]

k = a[0] 

print('K initial value = ', k) # prints: K initial value =  tensor([2.], requires_grad=True)
#Lots of code
very_complex_long_fn(a[0])
#Lots of code
print('K final value = ', k) # prints: K final value =  tensor([20.], requires_grad=True)
``` 
  
While it can be quite time consuming tracking down bugs involving a variable value (or type) changing unexpectedly, 
having an interpreter helps ensure it "crashes safely", by throwing an 
```Python AttributeError: 'SomeObj' object has no attribute 'some_attr'``` kind of error.

Most importantly, this is pretty much all Python can do since there is no check or compilation phase before
running the program. But in Rust, the idea is to prevent these kind of problems as much as possible during the compilation
phase, hence the ownership/borrowing rules.
 
> I'm not trying to bash on Python. It is an excellent language for what it was set out to do (hence the widespread 
> adoption). The point I'm trying to make here is that its values are very different from Rust's ones, so straight 
> forward Python (or even C/C++) -> Rust translation may (and sometimes should) not work at the risk of violating
> some of the core values Rust was built to provide.


### Prior Work 
 
Some people have tackled this problem before, such as [Rust-Autograd](https://github.com/raskr/rust-autograd) 
using lazily evaluated computation graphs and [Wyrm](https://github.com/maciejkula/wyrm) postponing the borrow checking
until runtime by using Rc<<RefCell<<T>>>>. 

Even then, as far as I known, they don't support indexing a tensor and 
using the indexed part of it in the computation graph, which is an essential (but very complicated) feature.  




 

 

