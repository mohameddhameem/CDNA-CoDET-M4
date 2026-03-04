import math

from sys import stdin, stdout





def solution(hc, hm, dc, dm, k, w, a):

    if k == 0:

        if math.ceil(hc/dm) >= math.ceil(hm/dc):

            return "YES"

        return "NO"

    for i in range(k+1):

        upgr_step_ch_kill = math.ceil((hc + i*a)/dm)

        upgr_step_mon_kill = math.ceil(hm/(dc+(k-i)*w))

        if upgr_step_mon_kill <= upgr_step_ch_kill:

            return "YES"

    return "NO"







test_cases = int(stdin.readline())



for it in range(test_cases):

    hc, dc, = (int(a) for a in stdin.readline().split())

    hm, dm = (int(a) for a in stdin.readline().split())

    k, w, a = (int(a) for a in stdin.readline().split())

    stdout.write(str(solution(hc, hm, dc, dm, k, w, a)) +'\n')

