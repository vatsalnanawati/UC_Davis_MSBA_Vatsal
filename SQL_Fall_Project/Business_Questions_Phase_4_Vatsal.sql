# Q1 - What is the default rate for loans across different age groups?

SELECT 
    CASE 
        WHEN p.person_age BETWEEN 18 AND 30 THEN '18-30'
        WHEN p.person_age BETWEEN 31 AND 45 THEN '31-45'
        WHEN p.person_age BETWEEN 46 AND 60 THEN '46-60'
        ELSE '60+'
    END AS age_group,
    COUNT(CASE WHEN l.loan_status = 'default' THEN 1 END) AS default_count,
    COUNT(*) AS total_loans,
    ROUND((COUNT(CASE WHEN l.loan_status = 'default' THEN 1 END) * 1.0 / COUNT(*)) * 100, 2) AS default_rate
FROM 
    PERSON p
JOIN 
    LOAN l ON p.person_id = l.person_id
GROUP BY 
    age_group
ORDER BY 
    age_group;

# Q2 - How does employment experience impact the average credit score of applicants?
SELECT 
    e.person_emp_exp AS employment_experience,
    ROUND(AVG(ch.credit_score), 2) AS avg_credit_score
FROM 
    EMPLOYMENT e
JOIN 
    CREDIT_HISTORY ch ON e.person_id = ch.person_id
GROUP BY 
    e.person_emp_exp
ORDER BY 
    e.person_emp_exp;
    
# Q3 - What are the top 3 most common loan intents among applicants, and what is their average loan amount?
SELECT 
    l.loan_intent,
    COUNT(*) AS loan_count,
    ROUND(AVG(lf.loan_amnt), 2) AS avg_loan_amount
FROM 
    LOAN l
JOIN 
    LOAN_FINANCIALS lf ON l.loan_id = lf.loan_id
GROUP BY 
    l.loan_intent
ORDER BY 
    loan_count DESC
LIMIT 3;

# Q4 - How does the percentage of income allocated to loans correlate with loan default rates?
SELECT 
    l.loan_percent_income,
    COUNT(CASE WHEN l.loan_status = 'default' THEN 1 END) AS default_count,
    COUNT(*) AS total_loans,
    ROUND((COUNT(CASE WHEN l.loan_status = 'default' THEN 1 END) * 1.0 / COUNT(*)) * 100, 2) AS default_rate
FROM 
    LOAN l
GROUP BY 
    l.loan_percent_income
ORDER BY 
    l.loan_percent_income;

# Q5 - Which credit score range has the highest proportion of defaults, and what are the average loan 
# interest rates in these ranges?
SELECT 
    CASE 
        WHEN ch.credit_score BETWEEN 300 AND 500 THEN '300-500'
        WHEN ch.credit_score BETWEEN 501 AND 700 THEN '501-700'
        ELSE '701+'
    END AS credit_score_range,
    COUNT(CASE WHEN l.loan_status = 'default' THEN 1 END) AS default_count,
    COUNT(*) AS total_loans,
    ROUND((COUNT(CASE WHEN l.loan_status = 'default' THEN 1 END) * 1.0 / COUNT(*)) * 100, 2) AS default_rate,
    ROUND(AVG(lf.loan_int_rate), 2) AS avg_interest_rate
FROM 
    CREDIT_HISTORY ch
JOIN 
    LOAN l ON ch.person_id = l.person_id
JOIN 
    LOAN_FINANCIALS lf ON l.loan_id = lf.loan_id
GROUP BY 
    credit_score_range
ORDER BY 
    credit_score_range;

# Q6 - For applicants with home ownership, what is the average loan amount and default rate compared to non-homeowners?
SELECT 
    ho.person_home_ownership,
    ROUND(AVG(lf.loan_amnt), 2) AS avg_loan_amount,
    COUNT(CASE WHEN l.loan_status = 'default' THEN 1 END) AS default_count,
    COUNT(*) AS total_loans,
    ROUND((COUNT(CASE WHEN l.loan_status = 'default' THEN 1 END) * 1.0 / COUNT(*)) * 100, 2) AS default_rate
FROM 
    HOME_OWNERSHIP ho
JOIN 
    LOAN l ON ho.person_id = l.person_id
JOIN 
    LOAN_FINANCIALS lf ON l.loan_id = lf.loan_id
GROUP BY 
    ho.person_home_ownership;

# Q7 - How does loan approval or rejection vary based on demographic factors like age, gender, and education level?
SELECT 
    p.person_age,
    p.person_gender,
    p.person_education,
    COUNT(CASE WHEN l.loan_status = 'approved' THEN 1 END) AS approvals,
    COUNT(CASE WHEN l.loan_status = 'rejected' THEN 1 END) AS rejections
FROM 
    PERSON p
JOIN 
    LOAN l ON p.person_id = l.person_id
GROUP BY 
    p.person_age, p.person_gender, p.person_education
ORDER BY 
    p.person_age, p.person_gender, p.person_education;

# Q8 - How does the loan default rate vary across different combinations of credit score and loan intent, and what are the average loan amounts and interest rates for each combination?
SELECT 
    ch.credit_score,
    l.loan_intent,
    COUNT(CASE WHEN l.loan_status = 'default' THEN 1 END) AS default_count,
    COUNT(*) AS total_loans,
    ROUND(AVG(lf.loan_amnt), 2) AS avg_loan_amount,
    ROUND(AVG(lf.loan_int_rate), 2) AS avg_interest_rate
FROM 
    LOAN l
JOIN 
    CREDIT_HISTORY ch ON l.person_id = ch.person_id
JOIN 
    LOAN_FINANCIALS lf ON l.loan_id = lf.loan_id
GROUP BY 
    ch.credit_score, l.loan_intent
ORDER BY 
    ch.credit_score, l.loan_intent;


# Q9 - What are the patterns of default rates for applicants with specific combinations of credit score, loan amount, and previous loan defaults, and how does this influence the average interest rate offered?

SELECT 
    l.loan_percent_income,
    COUNT(CASE WHEN l.loan_status = 'default' THEN 1 END) AS default_count,
    COUNT(*) AS total_loans,
    ROUND((COUNT(CASE WHEN l.loan_status = 'default' THEN 1 END) * 1.0 / COUNT(*)) * 100, 2) AS default_rate,
    e.person_emp_exp,
    p.person_education,
    ho.person_home_ownership
FROM 
    LOAN l
JOIN 
    EMPLOYMENT e ON l.person_id = e.person_id
JOIN 
    PERSON p ON l.person_id = p.person_id
JOIN 
    HOME_OWNERSHIP ho ON l.person_id = ho.person_id
GROUP BY 
    l.loan_percent_income, e.person_emp_exp, p.person_education, ho.person_home_ownership;


# Q10
SELECT 
    p.person_id,
    h.person_home_ownership AS homeownership_status,
    SUM(lf.loan_amnt) AS total_loan_amount,
    RANK() OVER (
        PARTITION BY h.person_home_ownership 
        ORDER BY SUM(lf.loan_amnt) DESC
    ) AS loan_rank_within_homeownership
FROM 
    PERSON p
JOIN 
    LOAN l ON p.person_id = l.person_id
JOIN 
    LOAN_FINANCIALS lf ON l.loan_id = lf.loan_id
JOIN 
    HOME_OWNERSHIP h ON p.person_id = h.person_id
GROUP BY 
    p.person_id, h.person_home_ownership
ORDER BY 
    h.person_home_ownership, loan_rank_within_homeownership;


