theorem a_plus_b_b_plus_a (a b : ℕ) : a + b = b + a :=
begin
  sorry
end

theorem inequality_chain
(a b c d: ℕ) (h₀ : a ≤ b) (h₁ : b ≤ c) (h₂ : c ≤ d) : a ≤ d :=
begin
  apply trans,
  --apply h₀,
  --apply trans,
  --apply h₁,
  --exact h₂
--end

#check 0