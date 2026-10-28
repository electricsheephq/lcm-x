"""Deterministic review matrices for the non-date base-identity guard."""


def primary_cells():
    cells = {}
    Q = [
        ("What is the project status?", "My project status is green."),
        ("What is my current project status?", "My project status is green."),
        ("What is my favorite color?", "My favorite color is blue."),
        (
            "How many engineers do I directly lead on the platform team?",
            "I directly lead 7 engineers on the platform team.",
        ),
        (
            "How much did I spend in total on lunch and dinner?",
            "I spent $12 on lunch and $18 on dinner.",
        ),
        ("What is my dog's name?", "My dog's name is Rex."),
        ("Which books have I read?", "I have read Dune and Emma."),
        ("What languages do I speak?", "I speak French and Spanish."),
        (
            "What was my most recent job title?",
            "My most recent job title is staff engineer.",
        ),
        (
            "How many hours did I sleep in total on Monday and Tuesday?",
            "I slept 7 hours on Monday and 8 hours on Tuesday.",
        ),
    ]
    N = [
        "I reviewed the plan on 2023-01-10 and took a vacation yesterday.",
        "I reviewed the plan on 2023-01-10 and again last week.",
        "I reviewed the plan on 2023-01-10 and again on 2023-01-12.",
        "I reviewed the plan on 2023-01-10 and again on January 12, 2023.",
        "We met on January 3, 2023 and again last week.",
        "We met on January 3, 2023 and again yesterday.",
        "We met on 2023-01-10, i.e. January 10, 2023.",
        "Plan dates: 2023-01/10, January 12, 2023 and February 3, 2023.",
        "We met on Jan 3rd, 2023 and again 3 days ago.",
        "I reviewed the plan on 2023-04-01 and took a vacation yesterday.",
        "I reviewed the plan on 2023-04-01 and again last week.",
    ]
    FILL = "Unrelated synthetic row."

    def whole(t):
        return [0, len(t)]

    for qi, (q, fact) in enumerate(Q):
        for ni, noise in enumerate(N):
            for place in ["separate", "prefix", "suffix", "whole"]:
                for n in (1, 3):
                    for carried in ("none", "noise", "fact"):
                        for qd in (None, "2023-03-22"):
                            rows = []
                            refs = []
                            if place == "separate":
                                rows.append(fact)
                                refs.append([0, 0, len(fact)])
                                fact_row = 0
                                extra = n
                            else:
                                if place == "prefix":
                                    row = noise + " " + fact
                                    s = len(noise) + 1
                                    e = len(row)
                                elif place == "suffix":
                                    row = fact + " " + noise
                                    s = 0
                                    e = len(fact)
                                else:
                                    row = noise + " " + fact
                                    s = 0
                                    e = len(row)
                                rows.append(row)
                                refs.append([0, s, e])
                                fact_row = 0
                                extra = n - 1
                            noise_rows = []
                            for k in range(extra):
                                rows.append(noise)
                                refs.append([len(rows) - 1, 0, len(noise)])
                                noise_rows.append(len(rows) - 1)
                            rows.append(FILL)
                            if carried == "none":
                                ci = len(rows) - 1
                            elif carried == "fact":
                                ci = fact_row
                            else:
                                ci = noise_rows[0] if noise_rows else fact_row
                            if (
                                carried == "noise"
                                and not noise_rows
                                and place == "separate"
                            ):
                                continue
                            cid = f"nd|q{qi}|n{ni}|{place}|k{n}|{carried}|{qd}"
                            cells[cid] = dict(
                                rows=rows,
                                carried_index=ci,
                                refs=refs,
                                question=q,
                                qd=qd,
                            )
    # latest with two candidates, one mixed-date
    for qd in (None, "2023-03-22"):
        for carried in ("none", "a", "b"):
            for ni, noise in enumerate(N):
                a = noise + " My project status is red."
                b = "My project status is green."
                for order in ("ab", "ba"):
                    rows = [a, b] if order == "ab" else [b, a]
                    rows.append(FILL)
                    ci = {"none": 2, "a": rows.index(a), "b": rows.index(b)}[carried]
                    refs = [[0, 0, len(rows[0])], [1, 0, len(rows[1])]]
                    for q in (
                        "What is my current project status?",
                        "What is the project status?",
                    ):
                        cid = f"lat|n{ni}|{order}|{carried}|{qd}|{q}"
                        cells[cid] = dict(
                            rows=rows,
                            carried_index=ci,
                            refs=refs,
                            question=q,
                            qd=qd,
                            stamps=[
                                1676462400.0,
                                1676462400.0 + 86400
                                if order == "ab"
                                else 1676462400.0 - 86400,
                                1676462400.0,
                            ],
                        )
    return cells


def secondary_cells():
    cells = {}
    Q = [
        (
            "How many different countries have I visited?",
            "I have visited France and Japan.",
        ),
        (
            "How much more did I spend on dinner than on lunch?",
            "I spent $12 on lunch and $18 on dinner.",
        ),
        (
            "What was my previous job title?",
            "My previous job title was senior engineer.",
        ),
        (
            "What should I recommend for a weekend trip?",
            "I recommend Kyoto for a weekend trip.",
        ),
    ]
    N = [
        "I reviewed the plan on 2023-01-10 and took a vacation yesterday.",
        "I reviewed the plan on 2023-01-10 and again last week.",
        "I reviewed the plan on 2023-01-10 and again on 2023-01-12.",
        "I reviewed the plan on 2023-01-10 and again on January 12, 2023.",
        "We met on January 3, 2023 and again last week.",
        "We met on January 3, 2023 and again yesterday.",
        "We met on 2023-01-10, i.e. January 10, 2023.",
        "Plan dates: 2023-01/10, January 12, 2023 and February 3, 2023.",
        "We met on Jan 3rd, 2023 and again 3 days ago.",
        "I reviewed the plan on 2023-04-01 and took a vacation yesterday.",
        "I reviewed the plan on 2023-04-01 and again last week.",
    ]
    FILL = "Unrelated synthetic row."

    def whole(t):
        return [0, len(t)]

    for qi, (q, fact) in enumerate(Q):
        for ni, noise in enumerate(N):
            for place in ["separate", "prefix", "suffix", "whole"]:
                for n in (1, 3):
                    for carried in ("none", "noise", "fact"):
                        for qd in (None, "2023-03-22"):
                            rows = []
                            refs = []
                            if place == "separate":
                                rows.append(fact)
                                refs.append([0, 0, len(fact)])
                                fact_row = 0
                                extra = n
                            else:
                                if place == "prefix":
                                    row = noise + " " + fact
                                    s = len(noise) + 1
                                    e = len(row)
                                elif place == "suffix":
                                    row = fact + " " + noise
                                    s = 0
                                    e = len(fact)
                                else:
                                    row = noise + " " + fact
                                    s = 0
                                    e = len(row)
                                rows.append(row)
                                refs.append([0, s, e])
                                fact_row = 0
                                extra = n - 1
                            noise_rows = []
                            for k in range(extra):
                                rows.append(noise)
                                refs.append([len(rows) - 1, 0, len(noise)])
                                noise_rows.append(len(rows) - 1)
                            rows.append(FILL)
                            if carried == "none":
                                ci = len(rows) - 1
                            elif carried == "fact":
                                ci = fact_row
                            else:
                                ci = noise_rows[0] if noise_rows else fact_row
                            if (
                                carried == "noise"
                                and not noise_rows
                                and place == "separate"
                            ):
                                continue
                            cid = f"nd|q{qi}|n{ni}|{place}|k{n}|{carried}|{qd}"
                            cells[cid] = dict(
                                rows=rows,
                                carried_index=ci,
                                refs=refs,
                                question=q,
                                qd=qd,
                            )
    return cells


def review_cells():
    return {
        f"{group}:{cid}": cell
        for group, cells in (("nd", primary_cells()), ("nd2", secondary_cells()))
        for cid, cell in cells.items()
    }


STAMP = 1676462400.0


def compile_cell(root, cell):
    from hermes_lcm import tools
    from tests.harness import engine_context

    with engine_context(root, embeddings_enabled=False) as engine:
        stamps = cell.get("stamps") or [STAMP] * len(cell["rows"])
        ids = [
            engine._store.append(
                str(j),
                dict(role="user", content=text, timestamp=stamps[j], source="cli"),
            )
            for j, text in enumerate(cell["rows"])
        ]
        engine._store.add_message_relations(
            [[(ids[cell["carried_index"]], "carried_stamp", None, 0, STAMP)]]
        )
        if cell.get("session_dates"):
            engine._session_occurrence_dates = dict(cell["session_dates"])
        refs = [f"lcm:{ids[row]}:{start}-{end}" for row, start, end in cell["refs"]]
        before = engine._store.connection.total_changes
        raw = tools.lcm_compile_evidence(
            dict(
                mode="auto",
                question=cell["question"],
                question_date=cell["qd"],
                baseline_refs=refs,
                budgets={"max_retrieval_calls": 0},
            ),
            engine=engine,
        )
        assert before == engine._store.connection.total_changes
        return raw
