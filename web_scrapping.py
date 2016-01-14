# Scraping

from bs4 import BeautifulSoup
from urllib2 import urlopen

def get_text(link_list):
    links = []
    text_list = list()
    for link in link_list:
        try:
            html = urlopen(link).read()
            soup = BeautifulSoup(html, 'lxml')
            text_list.append(''.join(str(soup.findAll('p'))))
            links.append(link)
        except:
            print link           
    return (links, text_list)


links = ['https://en.wikipedia.org/wiki/100_Bullets',
        'https://en.wikipedia.org/wiki/2000_AD_(comics)',
        'https://en.wikipedia.org/wiki/300_(comics)',
        'https://en.wikipedia.org/wiki/A_Contract_with_God',
        'https://en.wikipedia.org/wiki/Akira_(manga)',
        'https://en.wikipedia.org/wiki/All-Star_Superman',
        'https://en.wikipedia.org/wiki/Annihilation_(comics)',
        'https://en.wikipedia.org/wiki/Arkham_Asylum:_A_Serious_House_on_Serious_Earth',
        'https://en.wikipedia.org/wiki/Astonishing_X-Men',
        'https://en.wikipedia.org/wiki/Aya_of_Yop_City',
        'https://en.wikipedia.org/wiki/Barefoot_Gen',
        'https://en.wikipedia.org/wiki/Batman:_The_Killing_Joke',
        'https://en.wikipedia.org/wiki/Batman:_The_Long_Halloween',
        'https://en.wikipedia.org/wiki/Batman:_Year_One',
        'https://en.wikipedia.org/wiki/Black_Hole_(comics)',
        'https://en.wikipedia.org/wiki/Blankets_(comics)',
        'https://en.wikipedia.org/wiki/Bone_(comics)',
        'https://en.wikipedia.org/wiki/Born_Again_(comics)',
        'https://en.wikipedia.org/wiki/Chew_(comics)',
        'https://en.wikipedia.org/wiki/Civil_War_(comics)',
        'https://en.wikipedia.org/wiki/DMZ_(comics)',
        'https://en.wikipedia.org/wiki/Ex_Machina_(comics)',
        'https://en.wikipedia.org/wiki/Fables_(comics)',
        'https://en.wikipedia.org/wiki/From_Hell',
        'https://en.wikipedia.org/wiki/Fun_Home',
        'https://en.wikipedia.org/wiki/Ghost_World',
        'https://en.wikipedia.org/wiki/Girl_Genius',
        'https://en.wikipedia.org/wiki/Hellblazer',
        'https://en.wikipedia.org/wiki/Hellboy:_Seed_of_Destruction',
        'https://en.wikipedia.org/wiki/Kingdom_Come_(comics)',
        'https://en.wikipedia.org/wiki/Krazy_Kat',
        'https://en.wikipedia.org/wiki/List_of_Criminal_story_arcs',
        'https://en.wikipedia.org/wiki/List_of_Preacher_story_arcs',
        'https://en.wikipedia.org/wiki/Locke_%26_Key',
        'https://en.wikipedia.org/wiki/Lone_Wolf_and_Cub',
        'https://en.wikipedia.org/wiki/Louis_Riel_(comics)',
        'https://en.wikipedia.org/wiki/MIND_MGMT',
        'https://en.wikipedia.org/wiki/Marvels',
        'https://en.wikipedia.org/wiki/Maus',
        'https://en.wikipedia.org/wiki/Palestine_(comics)',
        'https://en.wikipedia.org/wiki/Persepolis_(comics)',
        'https://en.wikipedia.org/wiki/Powers_(comics)',
        'https://en.wikipedia.org/wiki/Saga_(comic_book)',
        'https://en.wikipedia.org/wiki/Scalped',
        'https://en.wikipedia.org/wiki/Scott_Pilgrim',
        'https://en.wikipedia.org/wiki/Sin_City',
        'https://en.wikipedia.org/wiki/Superman:_Red_Son',
        'https://en.wikipedia.org/wiki/Swamp_Thing',
        'https://en.wikipedia.org/wiki/The_Authority',
        'https://en.wikipedia.org/wiki/The_Dark_Knight_Returns',
        'https://en.wikipedia.org/wiki/The_Dark_Phoenix_Saga',
        'https://en.wikipedia.org/wiki/The_Galactus_Trilogy',
        'https://en.wikipedia.org/wiki/The_Invisibles',
        'https://en.wikipedia.org/wiki/The_League_of_Extraordinary_Gentlemen_(comics)',
        'https://en.wikipedia.org/wiki/The_Maxx',
        'https://en.wikipedia.org/wiki/The_New_Avengers_(comics)',
        'https://en.wikipedia.org/wiki/The_Night_Gwen_Stacy_Died',
        'https://en.wikipedia.org/wiki/The_Sandman_(Vertigo)',
        'https://en.wikipedia.org/wiki/The_Ultimates_(comic_book)',
        'https://en.wikipedia.org/wiki/The_Walking_Dead_(comic_book)',
        'https://en.wikipedia.org/wiki/Time_(xkcd)',
        'https://en.wikipedia.org/wiki/Transmetropolitan',
        'https://en.wikipedia.org/wiki/Uncanny_X-Men',
        'https://en.wikipedia.org/wiki/V_for_Vendetta',
        'https://en.wikipedia.org/wiki/Wanted_(comics)',
        'https://en.wikipedia.org/wiki/Watchmen',
        'https://en.wikipedia.org/wiki/Y:_The_Last_Man',
        ]

links, comic_text = get_text(links)