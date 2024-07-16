--import data.real.basic
--import data.nat.factorial.basic

theorem a_plus_b_b_plus_a
(a b : ℕ) : a + b = b + a :=
begin
  sorry
end

variables x y z : ℕ
variables H₀ : x ≤ y 
variables H₁ : y ≤ z 
#check trans
#check (trans H₀)
#check (trans H₀ H₁)

/-
theorem inequality_chain
(a b c d: ℕ) : a ≤ b → b ≤ c → c ≤ d → a ≤ d :=
begin
  intro h₀,
  intro h₁,
  intro h₂,
  apply trans,
end
-/

theorem inequality_chain
(a b c d: ℕ) (h₀ : a ≤ b) (h₁ : b ≤ c) (h₂ : c ≤ d) : a ≤ d :=
begin
  apply trans,
  --apply h₀,
  --apply trans,
  --apply h₁,
  --exact h₂
end

theorem inequality_chain
(a b c d: ℕ) (h₀ : a ≤ b) (h₁ : b ≤ c) (h₂ : c ≤ d) : a ≤ d :=
begin
  apply trans h₀,
  apply trans h₁,
  apply trans h₂,
  apply refl
  --apply h₀,
  --apply trans,
  --apply h₁,
  --exact h₂
end

/-

theorem mod_arith_1
(x : ℕ) : x % 2 = 0 → (x * x) % 2 = 0 :=
begin
  sorry
end

theorem n_less_2_pow_n
  (n : ℕ)
  (h₀ : 1 ≤ n) :
  n < 2^n :=
begin
  sorry
end

theorem a_plus_zero: ∀ (a : ℕ), a + 0 = a :=
begin
  sorry
end

theorem mathd_algebra_478
  (b h v : ℝ)
  (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)
  (h₁ : v = 1 / 3 * (b * h))
  (h₂ : b = 30)
  (h₃ : h = 13 / 2) :
  v = 65 :=
begin
  sorry
end

theorem ab_square:
∀ (a b: ℝ), (a + b)^2 = a^2 + b^2 + 2*a*b :=
begin
  sorry
end

-/