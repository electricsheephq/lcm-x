"""R-fix3: labeled object regressions and real suppress-mode decoy controls."""
import json
from pathlib import Path

import pytest

from test_retraction_delivery import add, deliver, detector, links
from test_recall_ranking_contract import corpus as corpus

# Labels compare d838da0f; synthetic rows only. B1–B4 are the approved changes.
CASES = [
    ('regression', 'B1-00', 'Forget what I told you about the 2.5 mg dose.', 'the 2.5 mg dose', 'The record is the 2.5 mg dose.', 'Room 2 is free.'),
    ('regression', 'B1-01', 'Forget what I told you about version 3.1 of the app.', 'version 3.1 of the app', 'The record is version 3.1 of the app.', 'Version 3 shipped yesterday.'),
    ('regression', 'B1-02', 'Forget what I told you about Dr. Patel.', 'dr. patel', 'The record is Dr. Patel.', 'Dr Who is on TV.'),
    ('regression', 'B1-03', 'Forget what I told you about St. Louis.', 'st. louis', 'The record is St. Louis.', 'St Patrick has a parade.'),
    ('regression', 'B1-04', 'Forget what I told you about the U.S. trip.', 'the u.s. trip', 'The record is the U.S. trip.', 'U turn ahead.'),
    ('regression', 'B1-05', 'Forget what I told you about bob@example.com.', 'bob@example.com', 'The record is bob@example.com.', 'Bob uses example.org.'),
    ('regression', 'B1-06', 'Forget what I told you about the $1,200 rent.', 'the $1,200 rent', 'The record is the $1,200 rent.', 'Gate 1 opens at nine.'),
    ('regression', 'B1-07', 'Forget what I told you about the 2.5 mg dose.', 'the 2.5 mg dose', 'The 2.5 mg dose works best.', 'Room 2 is free.'),
    ('regression', 'B1-08', 'Forget what I told you concerning the 3.5 GPA.', 'the 3.5 gpa', 'My GPA was 3.5 last term.', 'I have 3 kids.'),
    ('regression', 'B1-09', 'Forget what I told you about Dr. Patel.', 'dr. patel', 'Dr. Patel is my dentist.', 'Ask the Dr Who fans.'),
    ('regression', 'B1-10', 'Forget what I mentioned regarding the U.S. visa.', 'the u.s. visa', 'The U.S. visa expires soon.', 'U turn ahead.'),
    ('regression', 'B1-11', 'Forget my earlier statement about Mt. Fuji.', 'mt. fuji', 'Mt. Fuji was the trip goal.', 'The mt bike is broken.'),
    ('regression', 'B1-12', 'Disregard what I said about the $1,200 rent.', 'the $1,200 rent', 'The $1,200 rent is due.', 'Gate 1 opens at nine.'),
    ('regression', 'B1-13', 'Forget what I told you about the 10,000 steps goal.', 'the 10,000 steps goal', 'My goal is 10,000 steps.', 'Turn to page 10.'),
    ('regression', 'B1-14', 'Forget what I told you about my email jo.lee@mail.com.', 'my email jo.lee@mail.com', 'My email is jo.lee@mail.com now.', 'Jo sent an email.'),
    ('regression', 'B1-15', 'Forget what I told you about Python 3.12.', 'python 3.12', 'I use Python 3.12 daily.', 'Python 3 is old.'),
    ('regression', 'B1-16', 'Forget what I told you about Dr. Smith.', 'dr. smith', 'Dr. Smith is my cardiologist.', 'Dr. Jones moved offices.'),
    ('regression', 'B1-17', 'Forget what I told you about Mrs. Patel.', 'mrs. patel', 'Mrs. Patel lives next door.', 'Mrs. Garcia sold her car.'),
    ('regression', 'B1-18', 'Forget what I told you about St. Louis.', 'st. louis', 'St. Louis is where I grew up.', "St. Patrick's day parade was fun."),
    ('regression', 'B1-19', 'Forget what I told you about the 3.5 mile run.', 'the 3.5 mile run', 'The 3.5 mile run is on Sunday.', 'I have 3 kids.'),
    ('regression', 'B1-20', 'Forget what I said about the $4.50 coffee.', 'the $4.50 coffee', 'The $4.50 coffee is my daily habit.', 'I owe 4 dollars.'),
    ('regression', 'B1-21', 'Forget what I told you about example.com.', 'example.com', 'My site is example.com.', 'That was an example of bad design.'),
    ('regression', 'B1-22', 'Disregard what I mentioned regarding version 2.1 of the app.', 'version 2.1 of the app', 'Version 2.1 of the app crashed.', 'Version 2 shipped last year.'),
    ('regression', 'B1-23', 'Forget my earlier statement about Mt. Fuji.', 'mt. fuji', 'Mt. Fuji trip is in May.', 'Mt. Rainier was cloudy.'),
    ('regression', 'B1-24', 'Forget what I told you about Dr. Smith, it was wrong.', 'dr. smith', 'Dr. Smith is my cardiologist.', 'Dr. Jones moved offices.'),
    ('regression', 'B2-00', 'Forget what I told you earlier about my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-01', 'Disregard what I said earlier regarding my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-02', 'Forget what I told you before about my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-03', 'Disregard what I said before regarding my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-04', 'Forget what I told you yesterday about my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-05', 'Disregard what I said yesterday regarding my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-06', 'Forget what I told you last week about my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-07', 'Disregard what I said last week regarding my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-08', 'Forget what I told you a minute ago about my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-09', 'Disregard what I said a minute ago regarding my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-10', 'Forget what I told you earlier today about my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-11', 'Disregard what I said earlier today regarding my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-12', 'Forget what I told you the other day about my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-13', 'Disregard what I said the other day regarding my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-14', 'Forget what I told you so far about my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-15', 'Disregard what I said so far regarding my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-16', 'Forget what I just told you, about my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-17', 'Forget what I said to Bob about my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-18', 'Forget my statement from yesterday about my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-19', 'Forget what I mentioned so far concerning my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-20', 'Forget my earlier statement yesterday about my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('control', 'B2-21', 'Forget what I told you about my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('control', 'B2-22', 'Forget what I said regarding my peanut allergy.', 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B2-23', "Forget my statement's tone about my peanut allergy.", 'my peanut allergy', 'My peanut allergy is severe.', 'My favorite color is teal.'),
    ('regression', 'B3-00', 'Forget what I told you about Tom and Jerry.', 'tom and jerry', 'I watch Tom and Jerry cartoons.', 'Tom fixed my sink on Monday.'),
    ('regression', 'B3-01', 'Forget what I told you about salt and pepper shakers.', 'salt and pepper shakers', 'The salt and pepper shakers are blue.', 'Use less salt in the soup.'),
    ('regression', 'B4-00', 'Forget what I said about saffron tea, he said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-01', 'Forget what I said about saffron tea, my mom said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-02', 'Forget my earlier statement about saffron tea, he said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-03', 'Forget what I mentioned about saffron tea, my mom said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-04', 'Forget what I said about saffron tea, she wrote.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-05', 'Forget what I said about saffron tea, they replied.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-06', 'Forget what I said about saffron tea, said my sister.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-07', 'Forget what I mentioned about saffron tea, he said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-08', 'Forget what I mentioned about saffron tea, she wrote.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-09', 'Forget what I mentioned about saffron tea, they replied.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-10', 'Forget what I mentioned about saffron tea, said my sister.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'B4-11', 'Forget what I told you about saffron tea, he said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'B4-12', 'Forget what I told you about saffron tea, she wrote.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'B4-13', 'Forget what I told you about saffron tea, they replied.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'B4-14', 'Forget what I told you about saffron tea, my mom said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'B4-15', 'Forget what I told you about saffron tea, said my sister.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-16', 'Forget what I just said about saffron tea, he said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-17', 'Forget what I just said about saffron tea, she wrote.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-18', 'Forget what I just said about saffron tea, they replied.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-19', 'Forget what I just said about saffron tea, my mom said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-20', 'Forget what I just said about saffron tea, said my sister.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-21', 'Forget everything I mentioned about saffron tea, he said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-22', 'Forget everything I mentioned about saffron tea, she wrote.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-23', 'Forget everything I mentioned about saffron tea, they replied.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-24', 'Forget everything I mentioned about saffron tea, my mom said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-25', 'Forget everything I mentioned about saffron tea, said my sister.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'B4-26', 'Forget anything I told you about saffron tea, he said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'B4-27', 'Forget anything I told you about saffron tea, she wrote.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'B4-28', 'Forget anything I told you about saffron tea, they replied.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'B4-29', 'Forget anything I told you about saffron tea, my mom said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'B4-30', 'Forget anything I told you about saffron tea, said my sister.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-31', 'Forget my statement about saffron tea, he said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-32', 'Forget my statement about saffron tea, she wrote.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-33', 'Forget my statement about saffron tea, they replied.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-34', 'Forget my statement about saffron tea, my mom said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-35', 'Forget my statement about saffron tea, said my sister.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-36', 'Forget my earlier information about saffron tea, he said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-37', 'Forget my earlier information about saffron tea, she wrote.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-38', 'Forget my earlier information about saffron tea, they replied.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-39', 'Forget my earlier information about saffron tea, my mom said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-40', 'Forget my earlier information about saffron tea, said my sister.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-41', 'Forget my previous details about saffron tea, he said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-42', 'Forget my previous details about saffron tea, she wrote.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-43', 'Forget my previous details about saffron tea, they replied.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-44', 'Forget my previous details about saffron tea, my mom said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-45', 'Forget my previous details about saffron tea, said my sister.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-46', 'Forget my detail about saffron tea, he said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-47', 'Forget my detail about saffron tea, she wrote.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-48', 'Forget my detail about saffron tea, they replied.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-49', 'Forget my detail about saffron tea, my mom said.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'B4-50', 'Forget my detail about saffron tea, said my sister.', None, 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'F1-00', 'Forget what I told you about saffron tea.', 'saffron tea', 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'F1-01', 'Please forget what I told you.', '', 'Saffron tea is my favorite drink.', 'Archived tail: .'),
    ('control', 'F1-02', 'Forget what I told you before.', '', 'Saffron tea is my favorite drink.', 'Archived tail:  before.'),
    ('control', 'F1-03', 'Forget what I told you yesterday.', '', 'Saffron tea is my favorite drink.', 'Archived tail:  yesterday.'),
    ('control', 'F1-04', 'Forget what I told you again.', '', 'Saffron tea is my favorite drink.', 'Archived tail:  again.'),
    ('control', 'F1-05', 'Forget what I told you, thanks.', '', 'Saffron tea is my favorite drink.', 'Archived tail: , thanks.'),
    ('control', 'F1-06', 'Forget what I told you, it was wrong.', '', 'Saffron tea is my favorite drink.', 'Archived tail: , it was wrong.'),
    ('control', 'F1-07', 'Forget what I told you earlier today.', '', 'Saffron tea is my favorite drink.', 'Archived tail:  earlier today.'),
    ('control', 'F1-08', 'Forget what I told you last week.', '', 'Saffron tea is my favorite drink.', 'Archived tail:  last week.'),
    ('control', 'F1-09', 'Forget what I told you a minute ago.', '', 'Saffron tea is my favorite drink.', 'Archived tail:  a minute ago.'),
    ('control', 'F1-10', 'Forget what I said before.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I said before.'),
    ('control', 'F1-11', 'Forget what I said yesterday.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I said yesterday.'),
    ('control', 'F1-12', 'Forget what I said again.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I said again.'),
    ('control', 'F1-13', 'Forget what I said, thanks.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I said, thanks.'),
    ('control', 'F1-14', 'Forget what I said, it was wrong.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I said, it was wrong.'),
    ('control', 'F1-15', 'Forget what I said earlier today.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I said earlier today.'),
    ('control', 'F1-16', 'Forget what I said last week.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I said last week.'),
    ('control', 'F1-17', 'Forget what I said a minute ago.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I said a minute ago.'),
    ('control', 'F1-18', 'Forget what I mentioned before.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I mentioned before.'),
    ('control', 'F1-19', 'Forget what I mentioned yesterday.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I mentioned yesterday.'),
    ('control', 'F1-20', 'Forget what I mentioned again.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I mentioned again.'),
    ('control', 'F1-21', 'Forget what I mentioned, thanks.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I mentioned, thanks.'),
    ('control', 'F1-22', 'Forget what I mentioned, it was wrong.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I mentioned, it was wrong.'),
    ('control', 'F1-23', 'Forget what I mentioned earlier today.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I mentioned earlier today.'),
    ('control', 'F1-24', 'Forget what I mentioned last week.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I mentioned last week.'),
    ('control', 'F1-25', 'Forget what I mentioned a minute ago.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget what I mentioned a minute ago.'),
    ('control', 'F1-26', 'Forget my earlier statement before.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget my earlier statement before.'),
    ('control', 'F1-27', 'Forget my earlier statement yesterday.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget my earlier statement yesterday.'),
    ('control', 'F1-28', 'Forget my earlier statement again.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget my earlier statement again.'),
    ('control', 'F1-29', 'Forget my earlier statement, thanks.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget my earlier statement, thanks.'),
    ('control', 'F1-30', 'Forget my earlier statement, it was wrong.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget my earlier statement, it was wrong.'),
    ('control', 'F1-31', 'Forget my earlier statement earlier today.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget my earlier statement earlier today.'),
    ('control', 'F1-32', 'Forget my earlier statement last week.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget my earlier statement last week.'),
    ('control', 'F1-33', 'Forget my earlier statement a minute ago.', '', 'Saffron tea is my favorite drink.', 'Archived tail: Forget my earlier statement a minute ago.'),
    ('control', 'F1-34', 'Forget what I told you about saffron tea, thanks.', 'saffron tea', 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'F1-35', 'Forget what I told you regarding saffron tea; use coffee.', 'saffron tea', 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('control', 'F1-36', 'Forget what I said about saffron tea;', 'saffron tea', 'Saffron tea is my favorite drink.', 'Call the plumber before noon.'),
    ('regression', 'break-00', 'Forget what I told you about saffron tea. Use coffee.', 'saffron tea. use coffee', 'The record includes saffron tea. use coffee', 'Saffron tea stains fabric.'),
    ('regression', 'break-01', 'Forget what I told you about saffron! tea?', 'saffron! tea', 'The record includes saffron! tea', 'Saffron costs extra.'),
    ('regression', 'break-02', 'Forget what I told you about saffron? tea!', 'saffron? tea', 'The record includes saffron? tea', 'Saffron costs extra.'),
    ('control', 'break-03', 'Forget what I told you about saffron tea concerning coffee.', 'saffron tea concerning coffee', 'The record includes saffron tea concerning coffee', 'Call the plumber before noon.'),
    ('regression', 'break-04', 'Forget what I told you about version 1;2.', 'version 1;2', 'The record includes version 1;2', 'Version 1 shipped yesterday.'),
]

@pytest.mark.parametrize("label,case_id,text,expected,target_text,decoy_text", CASES,
                         ids=[f"{r[0]}-{r[1]}" for r in CASES])
def test_fix3_objects(label, case_id, text, expected, target_text, decoy_text):
    assert detector().request_object(text) == expected


@pytest.mark.parametrize("label,case_id,text,expected,target_text,decoy_text", CASES,
                         ids=[f"{r[0]}-{r[1]}" for r in CASES])
def test_fix3_suppress_decoy(corpus, monkeypatch, label, case_id, text, expected, target_text, decoy_text):
    target = add(corpus, target_text)
    decoy = add(corpus, decoy_text)  # Previous user: explicit requests must NOT go deictic.
    if expected == "":
        # F1 controls may retract only the true predecessor and intervening assistant.
        target = add(corpus, target_text)
    assistant = add(corpus, "Acknowledged the prior record.", "assistant")
    add(corpus, text)
    hits = [target, decoy, assistant]
    groups, reason = links(corpus, hits)
    assert reason is None
    off = deliver(corpus, monkeypatch, hits, "off", limit=3)
    payload = json.loads(deliver(corpus, monkeypatch, hits, "suppress", limit=3))
    survivors = {h.get("store_id") for h in payload["hits"]}
    assert decoy["store_id"] in survivors, "unrelated decoy withdrawn"
    if expected is None:
        assert groups == []
        assert json.loads(off) == payload
        assert target["store_id"] in survivors
    else:
        assert len(groups) == 1
        assert set(groups[0][1]) == ({0, 2} if expected == "" else {0})
        assert target["store_id"] not in survivors
        assert len(groups[0][1]) <= 3


def test_fix3_control_noncanonical_1200_byte_identity():
    rows = json.loads((Path(__file__).parents[1] / "fixtures/retraction_fix3_noncanonical.json").read_text())
    assert len(rows) == 1200
    actual = [[text, detector().request_object(text)] for text, _ in rows]
    assert json.dumps(actual, ensure_ascii=False) == json.dumps(rows, ensure_ascii=False)
