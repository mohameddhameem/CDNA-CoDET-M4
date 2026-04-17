"""Identifier obfuscation helpers.

Notes
-----
Renames symbols with stable short hashes for Python and Java code.
"""

import re
import argparse


# ── FNV-1a Hash (32-bit) ──────────────────────────────────────────────────────

def hash8(name: str) -> str:
    """Map an identifier to 'h' + 8 hex characters via FNV-1a."""
    h = 2166136261  # FNV offset basis
    for ch in name.encode():
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF  # FNV prime, 32-bit wrap
    return "h" + format(h, "08x")


# ── Reserved Word Sets ────────────────────────────────────────────────────────

PYTHON_RESERVED = {
    # Language keywords
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
    # Common built-ins
    "self", "cls", "super", "object", "type",
    "print", "input", "range", "len", "open", "repr", "format",
    "int", "float", "str", "bool", "list", "dict", "set", "tuple",
    "bytes", "bytearray", "memoryview", "complex",
    "abs", "all", "any", "bin", "chr", "dir", "divmod", "enumerate",
    "eval", "exec", "filter", "getattr", "globals", "hasattr", "hash",
    "hex", "id", "isinstance", "issubclass", "iter", "locals", "map",
    "max", "min", "next", "oct", "ord", "pow", "reversed", "round",
    "setattr", "slice", "sorted", "sum", "vars", "zip",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration", "NotImplementedError",
    # Dunder names
    "__init__", "__main__", "__name__", "__str__", "__repr__",
    "__len__", "__iter__", "__next__", "__class__", "__dict__",
}

JAVA_RESERVED = {
    # Language keywords
    "abstract", "assert", "boolean", "break", "byte", "case", "catch",
    "char", "class", "const", "continue", "default", "do", "double",
    "else", "enum", "extends", "final", "finally", "float", "for",
    "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "package", "private",
    "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while",
    "true", "false", "null",
    # Common types and classes
    "String", "Integer", "Boolean", "Double", "Float", "Long", "Short",
    "Byte", "Character", "Object", "Number", "Math", "System",
    "StringBuilder", "StringBuffer",
    "List", "ArrayList", "LinkedList", "Map", "HashMap", "TreeMap",
    "Set", "HashSet", "TreeSet", "Iterator", "Optional",
    "Exception", "RuntimeException", "NullPointerException",
    "IllegalArgumentException", "Override", "Deprecated", "SuppressWarnings",
    # Common methods and fields
    "out", "err", "in", "println", "print", "printf", "format",
    "main", "args", "length", "size", "toString", "equals",
    "hashCode", "compareTo", "getValue", "setValue", "get", "set",
    "add", "remove", "contains", "isEmpty", "clear",
}


# ── Identifier Extraction ─────────────────────────────────────────────────────

_IDENT_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")


def extract_identifiers(code: str, reserved: set) -> set:
    """Return all non-reserved identifiers found in the source code."""
    return {
        m.group(1)
        for m in _IDENT_RE.finditer(code)
        if m.group(1) not in reserved
    }


# ── Obfuscation Logic ─────────────────────────────────────────────────────────

def obfuscate(code: str, reserved: set) -> tuple[str, dict]:
    """
    Obfuscate source code and return (obfuscated_code, symbol_map).
    Names are replaced longest-first to prevent partial matches on
    shorter substrings.
    """
    identifiers = extract_identifiers(code, reserved)
    symbol_map = {name: hash8(name) for name in identifiers}

    result = code
    for name in sorted(symbol_map, key=len, reverse=True):
        result = re.sub(rf"\b{re.escape(name)}\b", symbol_map[name], result)

    return result, symbol_map


# ── Language Entry Points ─────────────────────────────────────────────────────

def obfuscate_python(code: str) -> tuple[str, dict]:
    """Obfuscate Python source code."""
    return obfuscate(code, PYTHON_RESERVED)


def obfuscate_java(code: str) -> tuple[str, dict]:
    """Obfuscate Java source code."""
    return obfuscate(code, JAVA_RESERVED)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    """Parse CLI options and obfuscate a Python or Java source file."""
    parser = argparse.ArgumentParser(
        description="Replace variable/function names in Python or Java code with 8-char hash codes."
    )
    parser.add_argument("file", help="Path to the input source file")
    parser.add_argument(
        "-l", "--lang",
        choices=["python", "java"],
        default=None,
        help="Source language (auto-detected from file extension if omitted)",
    )
    parser.add_argument("-o", "--output", default=None, help="Output file path (prints to stdout if omitted)")
    parser.add_argument("--map", action="store_true", help="Print the symbol mapping table after obfuscation")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        code = f.read()

    # Auto-detect language from file extension
    lang = args.lang
    if lang is None:
        if args.file.endswith(".py"):
            lang = "python"
        elif args.file.endswith(".java"):
            lang = "java"
        else:
            parser.error("Cannot detect language automatically. Use -l to specify python or java.")

    fn = obfuscate_python if lang == "python" else obfuscate_java
    obfuscated, symbol_map = fn(code)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(obfuscated)
        print(f"Written to: {args.output}")
    else:
        print(obfuscated)

    if args.map:
        print("\n── Symbol Map ──")
        for orig, hashed in sorted(symbol_map.items()):
            print(f"  {orig:30s} → {hashed}")


# ── Inline Demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run CLI if arguments are provided, otherwise execute the built-in demo
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        python_example = """
def calculate_fibonacci(n):
    result_list = []
    first_number = 0
    second_number = 1
    for index in range(n):
        result_list.append(first_number)
        temp_value = first_number + second_number
        first_number = second_number
        second_number = temp_value
    return result_list

class DataProcessor:
    def __init__(self, input_data):
        self.raw_data = input_data
        self.processed = False

    def process_items(self):
        cleaned_data = []
        for item in self.raw_data:
            if item is not None:
                cleaned_data.append(item)
        self.processed = True
        return cleaned_data
"""

        java_example = """
public class DataProcessor {
    private int itemCount;
    private String userName;

    public DataProcessor(String userName) {
        this.userName = userName;
        this.itemCount = 0;
    }

    public int calculateTotal(int[] numbers) {
        int totalSum = 0;
        for (int currentValue : numbers) {
            totalSum += currentValue;
        }
        return totalSum;
    }
}
"""

        for label, code, fn in [
            ("Python example", python_example, obfuscate_python),
            ("Java example",   java_example,   obfuscate_java),
        ]:
            print(f"\n{'='*60}")
            print(f"[{label}] Obfuscation result")
            print("=" * 60)
            result, smap = fn(code)
            print(result)
            print("\n── Symbol Map ──")
            for orig, hashed in sorted(smap.items()):
                print(f"  {orig:30s} → {hashed}")
