<br>

# 🇺🇸 English Version

> **Based on the explanation by Professor [Samuel Fernando](https://www.linkedin.com/in/samuelfernando2030/)**
> *Senior AI/ML Engineer | Researcher | MSc Quantum Computing | LinkedIn Top Voice*

<br><br>


##  What is LoRA?

LoRA (Low-Rank Adaptation) is a foundational technique for **efficient fine-tuning** of large models. As pointed out by **Professor Samuel Fernando**, many crucial advances were happening long before the GenAI hype.

Published in 2021 for GPT-3, LoRA made it possible to adapt huge models without touching billions of parameters.

<br><br>


##  How it works

Traditional fine-tuning updates the entire weight matrix **W**.
LoRA changes this:

<br><br>


## ✔ The base matrix W is frozen

### ✔ Only a low-rank update ΔW is learned:

```text
W' = W + ΔW
ΔW = B * A
```


<br>

Where **A** and **B** are low-rank matrices:

<br>

* A: (r × k)
* B: (d × r)
* with r ≪ d and k.

<br><br>

##  Numerical Example

A 1000 × 1000 matrix has **1,000,000 parameters**.

<br>

### With LoRA you only update:

* A: r × 1000
* B: 1000 × r

If r = 8 → **only 16,000 parameters**.
A massive reduction with minimal performance loss.

<br><br>



## Core insight

The meaningful update after massive pretraining **lives in a very low-dimensional subspace**.
That’s why LoRA is so efficient.

<br><br>

## Impact on GenAI

The paper → library → industry standard → product cycle became incredibly short.
LoRA is now a cornerstone of modern PEFT techniques.

<br><br>

## 🎓 Academic relevance

As Professor [**Samuel Fernando**]() emphasizes, academic research remains indispensable — many breakthroughs predate the GenAI popularization.

