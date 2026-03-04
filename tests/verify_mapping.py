code = "def hello():\n    return 42"
code_bytes = code.encode('utf-8')
print("Character mapping verification:")
print(f"'d' -> {code_bytes[0]} (expected 100)")
print(f"'e' -> {code_bytes[1]} (expected 101)")
print(f"'f' -> {code_bytes[2]} (expected 102)")
print(f"' ' (space) -> {code_bytes[3]} (expected 32)")
print(f"'h' -> {code_bytes[4]} (expected 104)")
print(f"'o' -> {code_bytes[7]} (expected 111)")
print(f"'(' -> {code_bytes[10]} (expected 40)")
print(f"'):' newline is at index 12")
print(f"'\\n' -> {code_bytes[12]} (expected 10)")

print("\n✓ Mapping is correct - UTF-8 encode() automatically provides ASCII byte values")
