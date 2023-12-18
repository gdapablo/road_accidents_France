"""

Config file for Streamlit App

"""

from member import Member


TITLE = "My Awesome App"

TEAM_MEMBERS = [
    Member(
        name="Pablo Galán de Anta",
        linkedin_url="http://www.linkedin.com/in/pablo-gal%C3%A1n-297075150",
        github_url="https://github.com/gdapablo",
    ),
    Member(
        name="Jennifer Poehlsen",
        linkedin_url="http://linkedin.com/in/jennifer-poehlsen-0aa7a825/",
        github_url="https://github.com/jpoehlsen",
    ),
    Member(
        name="Ilinca Suciu",
        linkedin_url="http://www.linkedin.com/in/ili-s",
        github_url="https://github.com/ili-s",
    )
]

PROMOTION = "Bootcamp Data Scientist - December 2023"
