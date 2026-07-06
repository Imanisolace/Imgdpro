import streamlit as st
import time

st.set_page_config(page_title="Mersenne Prime Finder", page_icon="🔢")

st.title("🔢 Mersenne Prime Finder - Lucas-Lehmer Test")
st.write("Finds primes of the form $M_p = 2^p - 1$ where p is prime")

def is_prime_small(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def lucas_lehmer(p):
    if p == 2:
        return True
    m = (1 << p) - 1
    s = 4
    for i in range(p - 2):
        s = (s * s - 2) % m
    return s == 0

col1, col2 = st.columns(2)
with col1:
    max_p = st.number_input("Max exponent p to test", min_value=2, max_value=2000, value=200, step=10)
with col2:
    run_btn = st.button("Find Mersenne Primes", type="primary")

if run_btn:
    progress = st.progress(0)
    status = st.empty()
    results = []
    
    primes_to_test = [p for p in range(2, max_p + 1) if is_prime_small(p)]
    total = len(primes_to_test)
    
    for idx, p in enumerate(primes_to_test):
        status.text(f"Testing p = {p}... {idx+1}/{total}")
        if lucas_lehmer(p):
            m = (1 << p) - 1
            digits = len(str(m))
            results.append((p, digits))
            st.success(f"Found! p = {p}, M_p has {digits} digits")
        progress.progress((idx + 1) / total)
        time.sleep(0.01) # so UI updates
    
    status.text("Done!")
    st.write("### All Mersenne Primes Found:")
    st.dataframe(results, column_config={"0": "Exponent p", "1": "Digits in M_p"})
    
st.caption("Note: Testing p > 1000 gets slow in the browser. For huge primes, use GIMPS.")
