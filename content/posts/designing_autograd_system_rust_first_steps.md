---
title: "Designing an Autograd System with Rust - First Steps"
date: 2020-02-12T23:37:58-03:00
draft: false
---

# Why do it?

Well, there is certainly a gap in the ecosystem currently and at least
[some](https://github.com/rust-ml/discussion/issues/1) people 
are interested.

In theory, Rust can do whatever C/C++ does given enough effort. 
Since most of Pytorch/Tensorflow is C++/CUDA, at least the C++ part
 should be doable.

Also, I'm too naive to not try it, but as alluded 
in [a previous post](/posts/current_deep_learning_ecosystem_from_a_rust_developer_perspective/)
I'm aware that it might not be so easy to follow Rust's ownership and borrowing
rules while providing similar ergonomics as Pytorch.

> In this post I try to explore the problem space and some tentative implementations. If you think
> I'm going in the wrong direction and have a better idea of how to proceed, please, let me know.

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


# What experience do we want to provide?

### Define By Run 

Pytorch has quite ergonomic and flexible training loops. This arguably comes mainly from:

- Defining the computational graph by just operating on the tensors themselves as if they were
*"regular"* variables (instead of constructing it using a DSL as with Tensorflow 1.x). 

- Tensor eagerly evaluated, which contributes to the feeling of them being "*just variables*". 

This last point has the added benefit of allowing the user to print them at any given moment between computation 
 and during a panic the stacktrace normally points to the line where the problem is (in contrast to lazy
  evaluation where the problem might only show up when the expression is evaluated and not where it is created).

Ok, we definitely want that! 
  
### Expected Features

Just to keep in mind other must haves, here is a list:

 - Manual Tensor creation
 - Tensor creation from operation on one or more existing Tensors
 - Tensor indexing and operations on the indexed values ***(hard)***
 - Using same Tensor on multiple Ops ***(hard)***
 - Tensors to be used as optimizable parameters 
 - Backpropagation through the Tensors (their gradients calculated)
 - Way to updated the parameters ```x = x - 0.1*x_grad``` and reuse them



# Naive Implementation: Distributed Borrowing and Cell

The first observation is that:

> In order to be usable in multiple Ops, Tensors need to be passed by *immutable* reference (otherwise they can't be 
shared).


At the same time to back-propagate the gradients through the computation graph we need:
- Each Tensor to provide a way to access its "parent Op" (because only the Op knows how to set the gradients) 
- Each Op to access its operands somehow in order to set their gradients

Of course, in order for all this to work all relevant Tensors and Ops need to be kept alive.

Let's check an example:

```Rust
let x = Tensor::from(2.);
let y = Tensor::from(3.);
let z = &x * &y;
```
<img src="/post_images/designing_autograd_system_rust/d1.svg" width="300">


So here Z needs to hold a reference of some kind to X and Y. Since we want to allow X and Y to be used in another
operation in the graph (imagine if you could only index a tensor once!), this need to be an immutable reference.


However, during backpropagation we need to actually to mutate the Tensors gradient field, which would require a
mutable reference to the field itself. 


```Rust
let x = Tensor::from(2.);
let y = Tensor::from(3.);
let z = &x * &y;
z.backward();
```

<img src="/post_images/designing_autograd_system_rust/d2.svg" width="300">

Maybe we can get around it by wrapping the gradient field in a 
[Cell](https://doc.rust-lang.org/std/cell/struct.Cell.html), allowing internal mutability and at the same
time making sure nobody holds a reference to the gradient field itself (the Cell wrapper prevents it) while mutating it. 
 
The main problem with this implementation is that this lead to pretty much all Tensors borrowing from each 
other behind the scenes and I suspect "distributed borrowing" can lead to an unhappy borrow checker very fast. 

> Also, at the risk of premature optimization, Tensors are expected to be created and freed in a loop, 
 so memory fragmentation can be an issue. Let's not worry about that now.

Let's try to implement it anyway.

We start with the Tensor structure. To keep it simple, lets pretend it only holds a single f32 value (instead of 
an N dimensional array). 

It needs to keep track of which Operation created it in order to set its operands gradients, so it needs either to own
or have a reference to the Op that created it. Lets have it owned to keep it simple. 
  
```Rust
struct Tensor{
    val: f32,
    grad: f32,
    op: Option<Op>
}
```

The Op needs to keep track of its operands and set their gradient when necessary, but it can't own the operands 
otherwise we won't be able to reuse the same Tensor in multiple operations, so we make it hold a reference.

```Rust
struct Op<'a>{
    operands: [&'a Tensor; 2]
}
```

But now, we need to change the Tensor signature to keep track of this lifetime: 

```Rust
struct Tensor<'a>{
    val: f32,
    grad: f32,
    op: Option<Op<'a>>
}
```
 
But now the Op signature which depends on the Tensor one has to change too: 


```Rust
struct Op<'a, 'b>{
    operands: [&'a Tensor<'b>; 2]
}
```

Now the Tensor signature has to change again... Ok, so this leads to infinite lifetimes.

One could argue that making the two lifetimes of Op equal solves the problem:

```Rust
struct Op<'a>{
    operands: [&'a Tensor<'a>; 2]
}
struct Tensor<'a>{
    val: f32,
    grad: f32,
    op: Option<Op<'a>>
}
```

But now, as far as I know, we are saying that the reference to the Tensor that Op holds lives as long as the 
Tensor itself and all the Ops the Tensor itself holds, which forces the Tensors to have the exact same lifetime.
This can only happen if they are inside the same container (like an arena allocator).

For more about why having the same lifetime can be problematic, check 
[Simon Sapin post on arenas and dropcheck](https://exyr.org/2018/rust-arenas-vs-dropck/)

Ok, this seems interesting. This post is running quite long already, so next time we will investigate how arena 
allocation can help us, drawing heavily from [Rufflewind's Post](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)
on reverse mode automatic differentiation.
