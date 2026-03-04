def who_tested(n, tests):
    from collections import defaultdict

    test_count = defaultdict(int)
    
    for test in tests:
        for student in test:
            test_count[student] += 1
    
    max_tests = max(test_count.values())
    result = [student for student, count in test_count.items() if count == max_tests]

    result.sort()
    return result

if __name__ == "__main__":
    n = int(input())
    tests = [input().split() for _ in range(n)]
    result = who_tested(n, tests)
    print("\n".join(result))