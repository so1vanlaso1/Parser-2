# from NEW_logic_pipeline.logic_skeleton import FormulaSkeleton
# from NEW_logic_pipeline.skeleton_builder import build_skeleton


# def test_fact_skeleton():
#     skeleton = build_skeleton("P1", "John is certified.")

#     assert skeleton.kind == "FACT"
#     assert skeleton.body is not None
#     assert "John is certified" in skeleton.body.text


# def test_exists_skeleton():
#     skeleton = build_skeleton("P1", "At least one student has completed a course.")

#     assert skeleton.kind == "EXISTS"
#     assert skeleton.quantifier == "exists"
#     assert skeleton.body is not None
#     assert "student" in skeleton.body.text
#     assert "completed" in skeleton.body.text


# def test_forall_skeleton():
#     skeleton = build_skeleton("P1", "All students receive training.")

#     assert skeleton.kind == "FORALL"
#     assert skeleton.quantifier == "forall"
#     assert skeleton.antecedent is not None
#     assert skeleton.consequent is not None
#     assert "student" in skeleton.antecedent.text
#     assert "receive training" in skeleton.consequent.text


# def test_rule_skeleton():
#     skeleton = build_skeleton(
#         "P1",
#         "If a student did not submit the final report, then they did not receive course recognition.",
#     )

#     assert skeleton.kind == "RULE"
#     assert skeleton.antecedent is not None
#     assert skeleton.consequent is not None
#     assert "did not submit" in skeleton.antecedent.text
#     assert "did not receive" in skeleton.consequent.text


# def test_only_if_skeleton():
#     skeleton = build_skeleton(
#         "P1",
#         "A student graduates only if the student passes the final exam.",
#     )

#     assert skeleton.kind == "ONLY_IF_RULE"
#     assert skeleton.antecedent is not None
#     assert skeleton.consequent is not None
#     assert "graduates" in skeleton.antecedent.text
#     assert "passes the final exam" in skeleton.consequent.text
#     assert "only_if_direction" in skeleton.risk_flags


# def test_iff_skeleton():
#     skeleton = build_skeleton(
#         "P1",
#         "A student is eligible if and only if the student passes the exam.",
#     )

#     assert skeleton.kind == "IFF"
#     assert skeleton.left is not None
#     assert skeleton.right is not None
#     assert "eligible" in skeleton.left.text
#     assert "passes the exam" in skeleton.right.text


# def test_non_if_rule_skeleton():
#     skeleton = build_skeleton(
#         "P1",
#         "Passing Philosophy grants eligibility for the Quantum Physics lab.",
#     )

#     assert skeleton.kind == "NON_IF_RULE"
#     assert skeleton.antecedent is not None
#     assert skeleton.consequent is not None
#     assert "Passing Philosophy" in skeleton.antecedent.text
#     assert "eligibility" in skeleton.consequent.text


# def test_negative_non_if_rule_skeleton():
#     skeleton = build_skeleton("P1", "Lack of approval prevents laboratory access.")

#     assert skeleton.kind == "NON_IF_RULE"
#     assert skeleton.consequent is not None
#     assert skeleton.consequent.negation_hint is True


# def test_modal_skeleton():
#     skeleton = build_skeleton(
#         "P1",
#         "Every smart home device is not necessarily energy efficient.",
#     )

#     assert skeleton.kind == "MODAL"
#     assert "modal_not_necessarily" in skeleton.risk_flags
#     assert skeleton.needs_review is True


# def test_obligation_skeleton():
#     skeleton = build_skeleton("P1", "Wearing goggles is mandatory in science laboratories.")

#     assert skeleton.kind == "OBLIGATION_RULE"
#     assert any(flag.startswith("deontic") for flag in skeleton.risk_flags)


# def test_meta_nested_implication_skeleton():
#     skeleton = build_skeleton(
#         "P1",
#         "If passing the exam implies graduation, then students who pass are eligible.",
#     )

#     assert skeleton.kind == "META"
#     assert skeleton.formula_tree is not None
#     assert skeleton.formula_tree.type == "implies"
#     assert len(skeleton.formula_tree.children) == 2
#     assert _contains_type(skeleton.formula_tree, "forall") or _contains_type(
#         skeleton.formula_tree,
#         "implies",
#         skip_root=True,
#     )


# def test_meta_consequent_is_if_rule_skeleton():
#     skeleton = build_skeleton(
#         "P1",
#         (
#             "If there exists at least one student who attends tutorials, "
#             "then if a student does not ask questions, the student does not attend tutorials."
#         ),
#     )

#     assert skeleton.kind == "META"
#     assert skeleton.formula_tree is not None
#     assert skeleton.formula_tree.type == "implies"
#     assert _contains_type(skeleton.formula_tree.children[0], "exists")
#     assert _contains_type(skeleton.formula_tree.children[1], "implies")


# def test_unknown_skeleton():
#     skeleton = build_skeleton(
#         "P1",
#         "This premise is structurally unclear and cannot be safely parsed @@@",
#     )

#     assert skeleton.kind == "UNKNOWN"
#     assert skeleton.needs_review is True


# def _contains_type(
#     node: FormulaSkeleton,
#     node_type: str,
#     *,
#     skip_root: bool = False,
# ) -> bool:
#     if not skip_root and node.type == node_type:
#         return True
#     return any(_contains_type(child, node_type) for child in node.children)


"""
100 additional skeleton-builder tests.

These tests are designed to extend the existing tests for:
    NEW_logic_pipeline.skeleton_builder.build_skeleton

They intentionally cover surface variation, negation, only-if direction,
iff/biconditional statements, non-if causal rules, deontic/obligation
phrases, modal ambiguity, and META/nested implication patterns.
"""

import pytest

from NEW_logic_pipeline.logic_skeleton import FormulaSkeleton
from NEW_logic_pipeline.skeleton_builder import build_skeleton


# Count: 100 parameterized test cases.
SKELETON_CASES = [
    # ------------------------------------------------------------------
    # FACT cases: named constants / concrete assertions
    # ------------------------------------------------------------------
    pytest.param(
        "fact_alex_registered",
        "Alex is registered for the seminar.",
        "FACT",
        {"body": ["Alex", "registered"]},
        {},
        id="001_fact_alex_registered",
    ),
    pytest.param(
        "fact_mina_paid",
        "Mina has paid the annual fee.",
        "FACT",
        {"body": ["Mina", "paid"]},
        {},
        id="002_fact_mina_paid",
    ),
    pytest.param(
        "fact_tuan_certified",
        "Tuan is certified.",
        "FACT",
        {"body": ["Tuan", "certified"]},
        {},
        id="003_fact_tuan_certified",
    ),
    pytest.param(
        "fact_sarah_active",
        "Sarah has active student status.",
        "FACT",
        {"body": ["Sarah", "active student status"]},
        {},
        id="004_fact_sarah_active",
    ),
    pytest.param(
        "fact_kelvin_submitted",
        "Kelvin submitted the lab report on time.",
        "FACT",
        {"body": ["Kelvin", "submitted"]},
        {},
        id="005_fact_kelvin_submitted",
    ),
    pytest.param(
        "fact_linh_health",
        "Linh has health issues.",
        "FACT",
        {"body": ["Linh", "health issues"]},
        {},
        id="006_fact_linh_health",
    ),
    pytest.param(
        "fact_john_faculty",
        "Dr. John is a faculty member.",
        "FACT",
        {"body": ["Dr. John", "faculty member"]},
        {},
        id="007_fact_john_faculty",
    ),
    pytest.param(
        "fact_iruma_not_last_year",
        "Iruma is not a last-year student.",
        "FACT",
        {"body": ["Iruma", "not", "last-year student"]},
        {},
        id="008_fact_iruma_not_last_year",
    ),
    pytest.param(
        "fact_an_overdue",
        "An returned books overdue.",
        "FACT",
        {"body": ["An", "returned books overdue"]},
        {},
        id="009_fact_an_overdue",
    ),
    pytest.param(
        "fact_frieren_oisp",
        "Frieren is an OISP student.",
        "FACT",
        {"body": ["Frieren", "OISP student"]},
        {},
        id="010_fact_frieren_oisp",
    ),
    pytest.param(
        "fact_course_ch3002",
        "CH3002 is taught by Professor Y.",
        "FACT",
        {"body": ["CH3002", "Professor Y"]},
        {},
        id="011_fact_course_ch3002",
    ),
    pytest.param(
        "fact_library_open",
        "The library is open today.",
        "FACT",
        {"body": ["library", "open today"]},
        {},
        id="012_fact_library_open",
    ),

    # ------------------------------------------------------------------
    # EXISTS cases
    # ------------------------------------------------------------------
    pytest.param(
        "exists_student_course",
        "At least one student has completed a course.",
        "EXISTS",
        {"body": ["student", "completed a course"]},
        {"quantifier": "exists"},
        id="013_exists_student_course",
    ),
    pytest.param(
        "exists_certified_student",
        "There exists at least one student who is certified.",
        "EXISTS",
        {"body": ["student", "certified"]},
        {"quantifier": "exists"},
        id="014_exists_certified_student",
    ),
    pytest.param(
        "exists_recommendation",
        "There is at least one student who receives a recommendation letter.",
        "EXISTS",
        {"body": ["student", "recommendation letter"]},
        {"quantifier": "exists"},
        id="015_exists_recommendation",
    ),
    pytest.param(
        "exists_tutorials",
        "There exists a student who is attending tutorials.",
        "EXISTS",
        {"body": ["student", "attending tutorials"]},
        {"quantifier": "exists"},
        id="016_exists_tutorials",
    ),
    pytest.param(
        "exists_available_book",
        "There is at least one available book.",
        "EXISTS",
        {"body": ["available book"]},
        {"quantifier": "exists"},
        id="017_exists_available_book",
    ),
    pytest.param(
        "exists_gps_drone",
        "At least one drone has GPS navigation.",
        "EXISTS",
        {"body": ["drone", "GPS navigation"]},
        {"quantifier": "exists"},
        id="018_exists_gps_drone",
    ),
    pytest.param(
        "exists_ai_prediction",
        "There exists at least one AI model that can make predictions.",
        "EXISTS",
        {"body": ["AI model", "predictions"]},
        {"quantifier": "exists"},
        id="019_exists_ai_prediction",
    ),
    pytest.param(
        "exists_student_council",
        "There exists at least one student who is a member of the student council.",
        "EXISTS",
        {"body": ["student", "student council"]},
        {"quantifier": "exists"},
        id="020_exists_student_council",
    ),
    pytest.param(
        "exists_qualified_person",
        "There exists at least one qualified person.",
        "EXISTS",
        {"body": ["qualified person"]},
        {"quantifier": "exists"},
        id="021_exists_qualified_person",
    ),
    pytest.param(
        "exists_online_course",
        "Some students are enrolled in at least one online course.",
        "EXISTS",
        {"body": ["students", "online course"]},
        {"quantifier": "exists"},
        id="022_exists_online_course",
    ),
    pytest.param(
        "exists_submitted_application",
        "Someone has submitted a scholarship application.",
        "EXISTS",
        {"body": ["submitted", "scholarship application"]},
        {"quantifier": "exists"},
        id="023_exists_submitted_application",
    ),
    pytest.param(
        "exists_researcher",
        "At least one individual is a researcher.",
        "EXISTS",
        {"body": ["individual", "researcher"]},
        {"quantifier": "exists"},
        id="024_exists_researcher",
    ),

    # ------------------------------------------------------------------
    # FORALL cases
    # ------------------------------------------------------------------
    pytest.param(
        "forall_training",
        "All students receive training.",
        "FORALL",
        {"antecedent": ["student"], "consequent": ["receive training"]},
        {"quantifier": "forall"},
        id="025_forall_training",
    ),
    pytest.param(
        "forall_pedagogical",
        "Every student has pedagogical skills.",
        "FORALL",
        {"antecedent": ["student"], "consequent": ["pedagogical skills"]},
        {"quantifier": "forall"},
        id="026_forall_pedagogical",
    ),
    pytest.param(
        "forall_books_available",
        "Every book in the library is available.",
        "FORALL",
        {"antecedent": ["book"], "consequent": ["available"]},
        {"quantifier": "forall"},
        id="027_forall_books_available",
    ),
    pytest.param(
        "forall_drones_range",
        "Every drone has a long remote control range.",
        "FORALL",
        {"antecedent": ["drone"], "consequent": ["long remote control range"]},
        {"quantifier": "forall"},
        id="028_forall_drones_range",
    ),
    pytest.param(
        "forall_cloud_tested",
        "All cloud services are thoroughly tested.",
        "FORALL",
        {"antecedent": ["cloud services"], "consequent": ["thoroughly tested"]},
        {"quantifier": "forall"},
        id="029_forall_cloud_tested",
    ),
    pytest.param(
        "forall_people_punctual",
        "Everyone is punctual.",
        "FORALL",
        {"antecedent": ["Everyone"], "consequent": ["punctual"]},
        {"quantifier": "forall"},
        id="030_forall_people_punctual",
    ),
    pytest.param(
        "forall_students_projects",
        "All students participate in projects.",
        "FORALL",
        {"antecedent": ["students"], "consequent": ["participate in projects"]},
        {"quantifier": "forall"},
        id="031_forall_students_projects",
    ),
    pytest.param(
        "forall_programmers_practice",
        "All programmers should practice coding.",
        "FORALL",
        {"antecedent": ["programmers"], "consequent": ["practice coding"]},
        {"quantifier": "forall"},
        id="032_forall_programmers_practice",
    ),
    pytest.param(
        "forall_employees_token",
        "Everyone in the company has been assigned a security token.",
        "FORALL",
        {"antecedent": ["company"], "consequent": ["security token"]},
        {"quantifier": "forall"},
        id="033_forall_employees_token",
    ),
    pytest.param(
        "forall_iot_connected",
        "Every IoT device used in agriculture is connected to the network.",
        "FORALL",
        {"antecedent": ["IoT device"], "consequent": ["connected to the network"]},
        {"quantifier": "forall"},
        id="034_forall_iot_connected",
    ),
    pytest.param(
        "forall_students_lms",
        "All students have access to the learning management system.",
        "FORALL",
        {"antecedent": ["students"], "consequent": ["learning management system"]},
        {"quantifier": "forall"},
        id="035_forall_students_lms",
    ),
    pytest.param(
        "forall_university_researchers",
        "All individuals are researchers.",
        "FORALL",
        {"antecedent": ["individuals"], "consequent": ["researchers"]},
        {"quantifier": "forall"},
        id="036_forall_university_researchers",
    ),

    # ------------------------------------------------------------------
    # RULE cases
    # ------------------------------------------------------------------
    pytest.param(
        "rule_no_training_no_foundation",
        "If a student does not receive training, then they do not have a research foundation.",
        "RULE",
        {"antecedent": ["does not receive training"], "consequent": ["do not have a research foundation"]},
        {},
        id="037_rule_no_training_no_foundation",
    ),
    pytest.param(
        "rule_no_cert_no_teach",
        "If a student is not certified, then they cannot teach.",
        "RULE",
        {"antecedent": ["not certified"], "consequent": ["cannot teach"]},
        {},
        id="038_rule_no_cert_no_teach",
    ),
    pytest.param(
        "rule_study_understand",
        "If a student studies, then they will understand the material.",
        "RULE",
        {"antecedent": ["studies"], "consequent": ["understand the material"]},
        {},
        id="039_rule_study_understand",
    ),
    pytest.param(
        "rule_lms_recordings",
        "If a student does not have LMS access, then they cannot view lecture recordings.",
        "RULE",
        {"antecedent": ["does not have LMS access"], "consequent": ["cannot view lecture recordings"]},
        {},
        id="040_rule_lms_recordings",
    ),
    pytest.param(
        "rule_registered_classical",
        "If a student registered for Advanced Physics, then they must have passed Classical Mechanics.",
        "RULE",
        {"antecedent": ["registered for Advanced Physics"], "consequent": ["passed Classical Mechanics"]},
        {},
        id="041_rule_registered_classical",
    ),
    pytest.param(
        "rule_passed_data",
        "If a student passed Classical Mechanics, then they are allowed to take Data Structures.",
        "RULE",
        {"antecedent": ["passed Classical Mechanics"], "consequent": ["Data Structures"]},
        {},
        id="042_rule_passed_data",
    ),
    pytest.param(
        "rule_no_approval_no_access",
        "If a person lacks advisor approval, then the person cannot access the advanced lab.",
        "RULE",
        {"antecedent": ["lacks advisor approval"], "consequent": ["cannot access the advanced lab"]},
        {},
        id="043_rule_no_approval_no_access",
    ),
    pytest.param(
        "rule_competition_stress",
        "If students suffer from excessive competition, they will experience higher stress and burnout.",
        "RULE",
        {"antecedent": ["excessive competition"], "consequent": ["higher stress", "burnout"]},
        {},
        id="044_rule_competition_stress",
    ),
    pytest.param(
        "rule_assignment_f",
        "If a student gets an F for the assignment, then the student must redo the assignment.",
        "RULE",
        {"antecedent": ["F for the assignment"], "consequent": ["redo the assignment"]},
        {},
        id="045_rule_assignment_f",
    ),
    pytest.param(
        "rule_no_gpa_no_recommend",
        "If a student does not meet the GPA requirement, then they are not recommended.",
        "RULE",
        {"antecedent": ["does not meet the GPA requirement"], "consequent": ["not recommended"]},
        {},
        id="046_rule_no_gpa_no_recommend",
    ),
    pytest.param(
        "rule_submit_contract_member",
        "If a student has not submitted the group contract, then they are not considered an active group member.",
        "RULE",
        {"antecedent": ["not submitted", "group contract"], "consequent": ["not considered", "active group member"]},
        {},
        id="047_rule_submit_contract_member",
    ),
    pytest.param(
        "rule_gps_obstacle",
        "If a drone has GPS navigation, then it has obstacle avoidance.",
        "RULE",
        {"antecedent": ["GPS navigation"], "consequent": ["obstacle avoidance"]},
        {},
        id="048_rule_gps_obstacle",
    ),
    pytest.param(
        "rule_secure_interface",
        "If an e-commerce website does not have a secure payment system, then it does not have a user-friendly interface.",
        "RULE",
        {"antecedent": ["does not have a secure payment system"], "consequent": ["does not have a user-friendly interface"]},
        {},
        id="049_rule_secure_interface",
    ),
    pytest.param(
        "rule_active_recall",
        "If a student uses active recall, they retain more information than passive review.",
        "RULE",
        {"antecedent": ["uses active recall"], "consequent": ["retain more information"]},
        {},
        id="050_rule_active_recall",
    ),
    pytest.param(
        "rule_token_manager",
        "If a person is assigned a security token, then they are a manager.",
        "RULE",
        {"antecedent": ["security token"], "consequent": ["manager"]},
        {},
        id="051_rule_token_manager",
    ),
    pytest.param(
        "rule_completed_training_machinery",
        "If someone completes the safety training, then they are allowed to operate heavy machinery.",
        "RULE",
        {"antecedent": ["completes the safety training"], "consequent": ["operate heavy machinery"]},
        {},
        id="052_rule_completed_training_machinery",
    ),
    pytest.param(
        "rule_coding_error_f",
        "If a student does the assignment incorrectly, then the student gets an F for the coding section.",
        "RULE",
        {"antecedent": ["assignment incorrectly"], "consequent": ["F for the coding section"]},
        {},
        id="053_rule_coding_error_f",
    ),
    pytest.param(
        "rule_research_recognition",
        "If a student conducts research, they will receive academic recognition.",
        "RULE",
        {"antecedent": ["conducts research"], "consequent": ["academic recognition"]},
        {},
        id="054_rule_research_recognition",
    ),
    pytest.param(
        "rule_course_online_engaging",
        "If a course is offered online, then it is engaging.",
        "RULE",
        {"antecedent": ["offered online"], "consequent": ["engaging"]},
        {},
        id="055_rule_course_online_engaging",
    ),
    pytest.param(
        "rule_understands_cpp_oop",
        "If a programmer understands C++, then they understand object-oriented programming.",
        "RULE",
        {"antecedent": ["understands C++"], "consequent": ["object-oriented programming"]},
        {},
        id="056_rule_understands_cpp_oop",
    ),

    # ------------------------------------------------------------------
    # ONLY_IF_RULE cases
    # ------------------------------------------------------------------
    pytest.param(
        "only_if_graduate_project",
        "A student graduates only if the student completes the graduation project.",
        "ONLY_IF_RULE",
        {"antecedent": ["graduates"], "consequent": ["completes the graduation project"]},
        {"risk_flags": ["only_if_direction"]},
        id="057_only_if_graduate_project",
    ),
    pytest.param(
        "only_if_borrow_card",
        "A person can borrow books only if the person has a valid library card.",
        "ONLY_IF_RULE",
        {"antecedent": ["borrow books"], "consequent": ["valid library card"]},
        {"risk_flags": ["only_if_direction"]},
        id="058_only_if_borrow_card",
    ),
    pytest.param(
        "only_if_lab_cert",
        "A student may access the advanced lab only if the student completed safety certification.",
        "ONLY_IF_RULE",
        {"antecedent": ["access the advanced lab"], "consequent": ["completed safety certification"]},
        {"risk_flags": ["only_if_direction"]},
        id="059_only_if_lab_cert",
    ),
    pytest.param(
        "only_if_scholarship_exam",
        "A student qualifies for the scholarship only if the student passes the qualifying exam.",
        "ONLY_IF_RULE",
        {"antecedent": ["qualifies for the scholarship"], "consequent": ["passes the qualifying exam"]},
        {"risk_flags": ["only_if_direction"]},
        id="060_only_if_scholarship_exam",
    ),
    pytest.param(
        "only_if_register_prereq",
        "A student registers for advanced subjects only if the student completes prerequisite courses.",
        "ONLY_IF_RULE",
        {"antecedent": ["registers for advanced subjects"], "consequent": ["completes prerequisite courses"]},
        {"risk_flags": ["only_if_direction"]},
        id="061_only_if_register_prereq",
    ),
    pytest.param(
        "only_if_exam_lab_score",
        "A student is allowed to take the exam only if the lab score is at least 4.0.",
        "ONLY_IF_RULE",
        {"antecedent": ["allowed to take the exam"], "consequent": ["lab score", "4.0"]},
        {"risk_flags": ["only_if_direction"]},
        id="062_only_if_exam_lab_score",
    ),
    pytest.param(
        "only_if_research_foundation_training",
        "A student has a research foundation only if the student received training.",
        "ONLY_IF_RULE",
        {"antecedent": ["research foundation"], "consequent": ["received training"]},
        {"risk_flags": ["only_if_direction"]},
        id="063_only_if_research_foundation_training",
    ),
    pytest.param(
        "only_if_data_structures_classical",
        "A student can take Data Structures only if the student passed Classical Mechanics.",
        "ONLY_IF_RULE",
        {"antecedent": ["Data Structures"], "consequent": ["passed Classical Mechanics"]},
        {"risk_flags": ["only_if_direction"]},
        id="064_only_if_data_structures_classical",
    ),

    # ------------------------------------------------------------------
    # IFF cases
    # ------------------------------------------------------------------
    pytest.param(
        "iff_eligible_exam",
        "A student is eligible if and only if the student passes the exam.",
        "IFF",
        {"left": ["eligible"], "right": ["passes the exam"]},
        {},
        id="065_iff_eligible_exam",
    ),
    pytest.param(
        "iff_exam_component_scores",
        "A student is allowed to take the exam if and only if all component scores are positive.",
        "IFF",
        {"left": ["allowed to take the exam"], "right": ["component scores are positive"]},
        {},
        id="066_iff_exam_component_scores",
    ),
    pytest.param(
        "iff_access_certification",
        "A person has lab access if and only if the person completed safety certification.",
        "IFF",
        {"left": ["lab access"], "right": ["completed safety certification"]},
        {},
        id="067_iff_access_certification",
    ),
    pytest.param(
        "iff_recommended_gpa",
        "A student is recommended if and only if the GPA requirement is met.",
        "IFF",
        {"left": ["recommended"], "right": ["GPA requirement is met"]},
        {},
        id="068_iff_recommended_gpa",
    ),
    pytest.param(
        "iff_lms_online",
        "A student has LMS access if and only if the student is enrolled in an online course.",
        "IFF",
        {"left": ["LMS access"], "right": ["online course"]},
        {},
        id="069_iff_lms_online",
    ),
    pytest.param(
        "iff_well_catalogued_available",
        "A book is well-catalogued if and only if it is available.",
        "IFF",
        {"left": ["well-catalogued"], "right": ["available"]},
        {},
        id="070_iff_well_catalogued_available",
    ),
    pytest.param(
        "iff_secure_customer_support",
        "An e-commerce website provides customer support if and only if it has a secure payment system.",
        "IFF",
        {"left": ["customer support"], "right": ["secure payment system"]},
        {},
        id="071_iff_secure_customer_support",
    ),
    pytest.param(
        "iff_understands_cpp_loops",
        "A programmer understands C++ if and only if the programmer understands loops.",
        "IFF",
        {"left": ["understands C++"], "right": ["understands loops"]},
        {},
        id="072_iff_understands_cpp_loops",
    ),

    # ------------------------------------------------------------------
    # NON_IF_RULE cases
    # ------------------------------------------------------------------
    pytest.param(
        "non_if_passing_philosophy_quantum",
        "Passing Philosophy grants eligibility for the Quantum Physics lab.",
        "NON_IF_RULE",
        {"antecedent": ["Passing Philosophy"], "consequent": ["eligibility", "Quantum Physics lab"]},
        {},
        id="073_non_if_passing_philosophy_quantum",
    ),
    pytest.param(
        "non_if_lack_approval_access",
        "Lack of advisor approval prevents advanced lab access.",
        "NON_IF_RULE",
        {"antecedent": ["Lack of advisor approval"], "consequent": ["advanced lab access"]},
        {"consequent_negation_hint": True},
        id="074_non_if_lack_approval_access",
    ),
    pytest.param(
        "non_if_failure_philosophy_scholarship",
        "Failure to pass Philosophy disqualifies students from Scholarships.",
        "NON_IF_RULE",
        {"antecedent": ["Failure to pass Philosophy"], "consequent": ["Scholarships"]},
        {"consequent_negation_hint": True},
        id="075_non_if_failure_philosophy_scholarship",
    ),
    pytest.param(
        "non_if_thesis_dormitory",
        "Thesis proposals grant dormitory access.",
        "NON_IF_RULE",
        {"antecedent": ["Thesis proposals"], "consequent": ["dormitory access"]},
        {},
        id="076_non_if_thesis_dormitory",
    ),
    pytest.param(
        "non_if_research_dormitory",
        "Research project approval grants dormitory access.",
        "NON_IF_RULE",
        {"antecedent": ["Research project approval"], "consequent": ["dormitory access"]},
        {},
        id="077_non_if_research_dormitory",
    ),
    pytest.param(
        "non_if_no_housing_participation",
        "Loss of housing prevents seminar participation.",
        "NON_IF_RULE",
        {"antecedent": ["Loss of housing"], "consequent": ["seminar participation"]},
        {"consequent_negation_hint": True},
        id="078_non_if_no_housing_participation",
    ),
    pytest.param(
        "non_if_good_behavior_climate",
        "Public recognition of good behavior fosters a positive school climate.",
        "NON_IF_RULE",
        {"antecedent": ["Public recognition", "good behavior"], "consequent": ["positive school climate"]},
        {},
        id="079_non_if_good_behavior_climate",
    ),
    pytest.param(
        "non_if_restorative_justice",
        "Restorative justice approaches help students understand the impact of their actions.",
        "NON_IF_RULE",
        {"antecedent": ["Restorative justice"], "consequent": ["understand the impact"]},
        {},
        id="080_non_if_restorative_justice",
    ),
    pytest.param(
        "non_if_positive_reinforcement",
        "Positive reinforcement strategies encourage good behavior.",
        "NON_IF_RULE",
        {"antecedent": ["Positive reinforcement strategies"], "consequent": ["good behavior"]},
        {},
        id="081_non_if_positive_reinforcement",
    ),
    pytest.param(
        "non_if_extra_credit_total_score",
        "Extra credit can increase a student's total score by up to 2 points.",
        "NON_IF_RULE",
        {"antecedent": ["Extra credit"], "consequent": ["increase", "total score"]},
        {},
        id="082_non_if_extra_credit_total_score",
    ),

    # ------------------------------------------------------------------
    # OBLIGATION_RULE / deontic cases
    # ------------------------------------------------------------------
    pytest.param(
        "obligation_goggles",
        "Wearing goggles is mandatory in science laboratories.",
        "OBLIGATION_RULE",
        {"body": ["Wearing goggles", "mandatory"]},
        {"risk_prefixes": ["deontic"]},
        id="083_obligation_goggles",
    ),
    pytest.param(
        "obligation_students_review",
        "Every student must review before the exam.",
        "OBLIGATION_RULE",
        {"body": ["student", "must review"]},
        {"risk_prefixes": ["deontic"]},
        id="084_obligation_students_review",
    ),
    pytest.param(
        "obligation_assignment",
        "Students are required to submit the assignment by Friday.",
        "OBLIGATION_RULE",
        {"body": ["required", "submit the assignment"]},
        {"risk_prefixes": ["deontic"]},
        id="085_obligation_assignment",
    ),
    pytest.param(
        "obligation_safety_form",
        "Employees must sign the safety compliance form.",
        "OBLIGATION_RULE",
        {"body": ["Employees", "must sign", "safety compliance form"]},
        {"risk_prefixes": ["deontic"]},
        id="086_obligation_safety_form",
    ),
    pytest.param(
        "obligation_credits",
        "Full-time students must take at least 14 credits per semester.",
        "OBLIGATION_RULE",
        {"body": ["Full-time students", "must take", "14 credits"]},
        {"risk_prefixes": ["deontic"]},
        id="087_obligation_credits",
    ),
    pytest.param(
        "obligation_orientation",
        "All students are required to attend the orientation.",
        "OBLIGATION_RULE",
        {"body": ["students", "required", "orientation"]},
        {"risk_prefixes": ["deontic"]},
        id="088_obligation_orientation",
    ),

    # ------------------------------------------------------------------
    # MODAL / ambiguity cases
    # ------------------------------------------------------------------
    pytest.param(
        "modal_not_necessarily_energy",
        "Every smart home device is not necessarily energy efficient.",
        "MODAL",
        {"body": ["not necessarily", "energy efficient"]},
        {"risk_flags": ["modal_not_necessarily"], "needs_review": True},
        id="089_modal_not_necessarily_energy",
    ),
    pytest.param(
        "modal_not_necessarily_ready",
        "A student who passes the quiz is not necessarily ready for the final exam.",
        "MODAL",
        {"body": ["not necessarily", "ready"]},
        {"risk_flags": ["modal_not_necessarily"], "needs_review": True},
        id="090_modal_not_necessarily_ready",
    ),
    pytest.param(
        "modal_may_receive",
        "Students may receive extra credit for participating in optional discussions.",
        "MODAL",
        {"body": ["may receive", "extra credit"]},
        {"needs_review": True},
        id="091_modal_may_receive",
    ),
    pytest.param(
        "modal_can_potential",
        "A research assistant can potentially access restricted data.",
        "MODAL",
        {"body": ["can potentially", "restricted data"]},
        {"needs_review": True},
        id="092_modal_can_potential",
    ),

    # ------------------------------------------------------------------
    # META / nested formula cases
    # ------------------------------------------------------------------
    pytest.param(
        "meta_rule_implies_exists",
        "If not taking the test leads to not passing, then at least one student takes the test.",
        "META",
        {},
        {"formula_root": "implies", "tree_contains": ["implies", "exists"]},
        id="093_meta_rule_implies_exists",
    ),
    pytest.param(
        "meta_exists_implies_rule",
        "If there exists at least one student who attends tutorials, then if a student does not ask questions, the student does not attend tutorials.",
        "META",
        {},
        {"formula_root": "implies", "tree_contains": ["exists", "implies"]},
        id="094_meta_exists_implies_rule",
    ),
    pytest.param(
        "meta_forall_implies_forall",
        "If all students participate in projects, then contributing to research leads to self-study.",
        "META",
        {},
        {"formula_root": "implies", "tree_contains": ["forall", "implies"]},
        id="095_meta_forall_implies_forall",
    ),
    pytest.param(
        "meta_previous_statement",
        "If the previous statement is true, then everyone is qualified.",
        "META",
        {},
        {"formula_root": "implies"},
        id="096_meta_previous_statement",
    ),
    pytest.param(
        "meta_policy_enforces_rule",
        "The Research-Thesis policy enforces Philosophy-Research compliance.",
        "META",
        {},
        {"tree_contains": ["implies"]},
        id="097_meta_policy_enforces_rule",
    ),
    pytest.param(
        "meta_rule_ensures_universal",
        "The rule 'No Thesis implies No Housing' ensures all students submit Research.",
        "META",
        {},
        {"tree_contains": ["implies", "forall"]},
        id="098_meta_rule_ensures_universal",
    ),
    pytest.param(
        "meta_nested_consequent_rule",
        "If having a long remote control range implies obstacle avoidance, then if a drone has GPS navigation, it has obstacle avoidance.",
        "META",
        {},
        {"formula_root": "implies", "tree_contains": ["implies"]},
        id="099_meta_nested_consequent_rule",
    ),
    pytest.param(
        "meta_implication_to_universal",
        "If eligibility guarantees test-taking, then not passing the prerequisite guarantees not graduating.",
        "META",
        {},
        {"formula_root": "implies", "tree_contains": ["implies"]},
        id="100_meta_implication_to_universal",
    ),
]


@pytest.mark.parametrize("case_id,premise,expected_kind,field_checks,extra_checks", SKELETON_CASES)
def test_100_more_skeleton_cases(case_id, premise, expected_kind, field_checks, extra_checks):
    skeleton = build_skeleton(case_id, premise)

    assert skeleton.kind == expected_kind

    for field_name, snippets in field_checks.items():
        node = getattr(skeleton, field_name)
        _assert_node_contains(node, snippets)

    if "quantifier" in extra_checks:
        assert skeleton.quantifier == extra_checks["quantifier"]

    for flag in extra_checks.get("risk_flags", []):
        assert flag in skeleton.risk_flags

    for prefix in extra_checks.get("risk_prefixes", []):
        assert any(flag.startswith(prefix) for flag in skeleton.risk_flags)

    if "needs_review" in extra_checks:
        assert skeleton.needs_review is extra_checks["needs_review"]

    if extra_checks.get("consequent_negation_hint") is True:
        assert skeleton.consequent is not None
        assert skeleton.consequent.negation_hint is True

    if "formula_root" in extra_checks:
        assert skeleton.formula_tree is not None
        assert skeleton.formula_tree.type == extra_checks["formula_root"]

    for node_type in extra_checks.get("tree_contains", []):
        assert skeleton.formula_tree is not None
        assert _contains_type(skeleton.formula_tree, node_type)


def _assert_node_contains(node, snippets):
    assert node is not None
    node_text = getattr(node, "text", "")
    normalized = node_text.lower()
    for snippet in snippets:
        assert snippet.lower() in normalized


def _contains_type(
    node: FormulaSkeleton,
    node_type: str,
    *,
    skip_root: bool = False,
) -> bool:
    if not skip_root and node.type == node_type:
        return True
    return any(_contains_type(child, node_type) for child in node.children)
