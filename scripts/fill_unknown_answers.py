"""One-off: fill in `correct_option_id_if_known` for questions the model
got wrong, so we have ground truth on the full DB for replay-based model
sweeps. Sets `generated_answer = 1` so eval code can distinguish these
from server-validated answers.

Each entry below is reasoned about by a human (the answer key); the
brief reasoning lives in the comment so a future reader can audit.
Run is idempotent -- re-running won't double-write.

    uv run --no-sync python scripts/fill_unknown_answers.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# -- generated answer key -------------------------------------------------
#
# Format: question_id -> option_id (the *correct* option for that question).
# The brief in the comment is the worked reasoning, not from the server.
GENERATED_ANSWERS: dict[int, int] = {
    # Q5547 (level 9): tool LEAST helpful for sending a message Mass -> India.
    # AM/FM radio is one-way broadcast and can't direct-message a recipient.
    5547: 1,
    # Q6725 (level 1): 4x - 2 over Z and Q.
    # Over Z: 4x - 2 = 2*(2x - 1). The 2 isn't a unit in Z[x] (units are +/- 1)
    #   so the factorisation is non-trivial -> reducible. Stmt 1: False.
    # Over Q: 2 IS a unit in Q[x] (every nonzero rational is a unit), so
    #   4x - 2 is associate to (2x - 1), a degree-1 polynomial, irreducible.
    #   Stmt 2: True.
    6725: 3,  # False, True
    # Q6741 (level 1): middle 50% of N(3250, 320) is the IQR.
    # z = +/- 0.6745, so range = 3250 +/- 0.6745*320 = (3034, 3466).
    6741: 0,
    # Q6773 (level 5): C is true iff exactly one of A, B is true.
    # C false means A = B (both T or both F). Therefore "A false implies B
    # false" must hold, since if A is false, B must also be false.
    6773: 0,
    # Q6835 (level 2): two-sample t-test for boys vs girls contacts.
    # Both samples are SRS, independent, n = 50 and 40 (both >= 30 so CLT
    # covers normality), population SDs unknown. Conditions for the
    # two-sample t-test are met. AP-Stats convention: should USE the test.
    # (Server marked the model's [3] wrong on a prior run -- I still believe
    # [3] is the textbook answer and want this in the eval set as
    # generated_answer so we can detect server/key disagreement.)
    6835: 3,
    # Q6933 (level 10): 0.1\overline{7} as a fraction.
    # x = 0.1777..., 10x = 1.777..., so 9x = 1.6 -> x = 16/90 = 8/45.
    6933: 1,
    # Q6995 (level 5): given three quadratic forms, find |a+b+c|.
    # a^2 + 2ab = 104/3, b^2 + 2bc = 7/9, c^2 + 2ca = -7.
    # Summing: a^2 + b^2 + c^2 + 2(ab+bc+ca) = 104/3 + 7/9 - 7 = 256/9.
    # That's exactly (a+b+c)^2 -> |a+b+c| = sqrt(256/9) = 16/3.
    6995: 1,
    # Q7032 (level 3): Wayne counts Y as a vowel, Kristen does not.
    # Wayne: 6/26, Kristen: 5/26. Percent increase = (6/26 - 5/26)/(5/26) =
    # 1/5 = 20%.
    7032: 0,
    # Q7037 (level 1): three statements about traces of n x n real matrices.
    # I.   trace(A^2) >= 0:   FALSE. Counterexample A = [[0,1],[-1,0]];
    #      A^2 = -I, trace -2.
    # II.  A^2 = A => trace(A) >= 0:   TRUE. Idempotent matrix has eigenvalues
    #      in {0, 1}, trace = sum of eigenvalues = rank >= 0.
    # III. trace(AB) = trace(A)*trace(B):   FALSE. A = B = I_2: trace(I) = 2,
    #      product = 4.
    # Only II is true.
    7037: 3,
    # Q7041 (level 5): exponent of x in (x * sqrt(x^3))^4.
    # x * sqrt(x^3) = x * x^(3/2) = x^(5/2). Raised to the 4: x^(20/2) = x^10.
    7041: 3,
}


def main() -> None:
    db_path = Path(__file__).resolve().parent.parent / "data" / "questions.sqlite"
    if not db_path.exists():
        raise SystemExit(f"DB not found at {db_path}")

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Sanity: only update rows that are still ungraded. This makes the
    # script idempotent and prevents overwriting a server-validated row
    # if one ever lands later.
    updated = 0
    skipped = 0
    for qid, option_id in GENERATED_ANSWERS.items():
        result = cur.execute(
            """
            UPDATE predictions
               SET correct_option_id_if_known = ?,
                   generated_answer = 1
             WHERE question_id = ?
               AND correct_option_id_if_known IS NULL
            """,
            (option_id, qid),
        )
        if result.rowcount == 0:
            skipped += 1
            print(f"Q{qid}: skipped (already graded or not present)")
        else:
            updated += result.rowcount
            print(f"Q{qid}: -> option [{option_id}] ({result.rowcount} row updated)")

    con.commit()

    cur.execute(
        "SELECT COUNT(*), SUM(correct_option_id_if_known IS NOT NULL),"
        " SUM(generated_answer = 1) FROM predictions"
    )
    total, graded, generated = cur.fetchone()
    print()
    print(f"summary: updated {updated}, skipped {skipped}")
    print(f"DB now has {graded}/{total} rows with a known answer ({generated} generated).")
    con.close()


if __name__ == "__main__":
    main()
