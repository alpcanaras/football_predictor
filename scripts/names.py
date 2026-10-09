"""
Team-name matcher
=================
football-data.co.uk names clubs its own way ("Buyuksehyr" for İstanbul
Başakşehir, "Ein Frankfurt", "M'gladbach"), while coupons, bookmakers and
people use full names ("VfL Wolfsburg", "Başakşehir", "Man Utd"). This module
maps all of them onto the one name the models and data use (the *canonical*
name), and shows people the full name (the *display* name).

Resolution is deliberately layered, most to least certain:

  memory    a pairing you confirmed once ("buyuksehr" -> Buyuksehyr), kept in
            data/_toto/name_memory.json
  exact     the accent- and punctuation-folded name itself
  alias     a curated variant (TEAMS below)
  core      the same name once club-type words are dropped
            ("VfL Wolfsburg", "FC Augsburg", "AC Milan" -> Wolfsburg, ...)
  contains  the query is the canonical name plus extra words
            ("Newcastle United" -> Newcastle), only when that is unique
  fuzzy     a near-miss spelling ("Fenerbache"), only when it is the clear
            winner; callers show these as guesses, not facts

Anything ambiguous ("Madrid", "Paris") resolves to nothing and comes back as
suggestions instead: a wrong silent match puts the wrong team on a coupon.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import unicodedata
from dataclasses import dataclass

_TRANSLIT = str.maketrans({
    'ı': 'i', 'İ': 'i', 'ß': 'ss', 'ø': 'o', 'Ø': 'o', 'æ': 'ae', 'Æ': 'ae',
    'œ': 'oe', 'đ': 'd', 'ð': 'd', 'þ': 'th', 'ł': 'l', 'Ł': 'l',
})

# Words that say what kind of club it is, not which club. Dropped when
# comparing "cores" so "VfL Wolfsburg" and "Wolfsburg" meet. Deliberately
# short: "united", "city", "real", "sporting" tell clubs apart.
NOISE = frozenset("""
fc fk sk sc cf afc ac as us ss ssc vfl vfb tsg sv fsv tsv spvgg bsc rcd rc cd
ud sd ca ad club de da do del calcio sad kv krc kaa rsc ogc losc aj nk hnk fcv
kvc if bk ik ff pfc gnk kf
""".split())


def norm(s) -> str:
    """Lower-case, accent-free, punctuation-free: 'Beşiktaş JK' -> 'besiktas jk'."""
    s = str(s or '').translate(_TRANSLIT)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"['’`´]", '', s).replace('&', ' and ')
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return ' '.join(s.split())


def core(s) -> tuple[str, ...]:
    """Distinctive tokens of a name, in order."""
    toks = norm(s).split()
    kept = [t for t in toks if t not in NOISE]
    if len(kept) > 1:                      # '1 fc koln', 'mainz 05' -> drop numbers
        no_num = [t for t in kept if not t.isdigit()]
        kept = no_num or kept
    return tuple(kept or toks)


# =============================================================================
# CURATED NAMES   canonical (as in the data) -> (display name, aliases)
# =============================================================================
# Only clubs whose data name differs from the name people use need a row; the
# rest resolve by themselves. tests/test_names.py checks every key is a real
# name in the data, so a typo here fails loudly instead of matching nothing.
TEAMS: dict[str, tuple[str, str]] = {
    # --- Türkiye
    'Buyuksehyr': ('İstanbul Başakşehir', 'Basaksehir; Istanbul Basaksehir; Basaksehir FK; Buyuksehir; Buyuksehir Belediyespor; Medipol Basaksehir; Istanbul Buyuksehir'),
    'Besiktas': ('Beşiktaş', 'Besiktas JK; BJK; Kara Kartal'),
    'Fenerbahce': ('Fenerbahçe', 'Fenerbahce SK; Fener; FB'),
    'Galatasaray': ('Galatasaray', 'Gala; GS; Cimbom; Galatasaray SK'),
    'Trabzonspor': ('Trabzonspor', 'Trabzon'),
    'Goztep': ('Göztepe', 'Goztep SK'),
    'Genclerbirligi': ('Gençlerbirliği', 'Genclerbirligi SK; Genclerbirlik'),
    'Kasimpasa': ('Kasımpaşa', 'Kasimpasa SK'),
    'Karagumruk': ('Fatih Karagümrük', 'Fatih Karagumruk; Karagumruk SK'),
    'Rizespor': ('Çaykur Rizespor', 'Caykur Rizespor; Rize; Rizespor'),
    'Gaziantep': ('Gaziantep FK', 'Gaziantep; Gaziantep Futbol Kulubu'),
    'Corum': ('Çorum FK', 'Corum FK; Corumspor'),
    'Eyupspor': ('Eyüpspor', 'Eyup; Eyupspor'),
    'Konyaspor': ('Konyaspor', 'Konya; Tumosan Konyaspor'),
    'Kayserispor': ('Kayserispor', 'Kayseri'),
    'Samsunspor': ('Samsunspor', 'Samsun'),
    'Antalyaspor': ('Antalyaspor', 'Antalya'),
    'Alanyaspor': ('Alanyaspor', 'Alanya; Corendon Alanyaspor'),
    'Amedspor': ('Amedspor', 'Amed; Amed SK; Diyarbakir'),
    'Kocaelispor': ('Kocaelispor', 'Kocaeli'),
    'Erzurumspor': ('Erzurumspor FK', 'Erzurum; Erzurumspor'),
    'Ad. Demirspor': ('Adana Demirspor', 'Adana Demirspor; Adana'),
    'Sivasspor': ('Sivasspor', 'Sivas'),
    'Hatayspor': ('Hatayspor', 'Hatay'),
    # --- Germany
    'Bayern Munich': ('Bayern München', 'Bayern Munchen; FC Bayern; FC Bayern Munchen; FC Bayern Munich; Bayern; FCB'),
    'Dortmund': ('Borussia Dortmund', 'BVB; BVB 09; Borussia Dortmund'),
    'Ein Frankfurt': ('Eintracht Frankfurt', 'Frankfurt; SGE; Eintracht'),
    'FC Koln': ('1. FC Köln', 'Koln; Köln; Cologne; FC Cologne; 1 FC Koln'),
    "M'gladbach": ('Borussia Mönchengladbach', 'Gladbach; Monchengladbach; Borussia Monchengladbach; Borussia M Gladbach'),
    'Leverkusen': ('Bayer Leverkusen', 'Bayer 04 Leverkusen; Bayer Leverkusen; B04'),
    'Mainz': ('1. FSV Mainz 05', 'Mainz 05; FSV Mainz; FSV Mainz 05'),
    'Hamburg': ('Hamburger SV', 'HSV; Hamburg SV'),
    'Heidenheim': ('1. FC Heidenheim', 'FC Heidenheim; Heidenheim 1846; 1 FC Heidenheim'),
    'Hoffenheim': ('TSG Hoffenheim', '1899 Hoffenheim; TSG 1899 Hoffenheim'),
    'Union Berlin': ('1. FC Union Berlin', 'FC Union Berlin; 1 FC Union Berlin'),
    'Werder Bremen': ('Werder Bremen', 'Bremen; SV Werder Bremen; Werder'),
    'St Pauli': ('FC St. Pauli', 'FC St Pauli; Sankt Pauli; St. Pauli'),
    'Schalke 04': ('FC Schalke 04', 'Schalke'),
    'Stuttgart': ('VfB Stuttgart', 'Stuttgart'),
    'Wolfsburg': ('VfL Wolfsburg', 'Wolfsburg'),
    'Freiburg': ('SC Freiburg', 'Freiburg'),
    'Augsburg': ('FC Augsburg', 'Augsburg'),
    'Paderborn': ('SC Paderborn 07', 'Paderborn 07; Paderborn'),
    'RB Leipzig': ('RB Leipzig', 'Leipzig; Rasenballsport Leipzig'),
    'Elversberg': ('SV Elversberg', 'Elversberg'),
    'Bielefeld': ('Arminia Bielefeld', 'Arminia'),
    'Bochum': ('VfL Bochum', 'Bochum'),
    'Braunschweig': ('Eintracht Braunschweig', 'Braunschweig'),
    'Cottbus': ('Energie Cottbus', 'Cottbus'),
    'Darmstadt': ('SV Darmstadt 98', 'Darmstadt 98; Darmstadt'),
    'Dresden': ('Dynamo Dresden', 'Dresden'),
    'Fortuna Dusseldorf': ('Fortuna Düsseldorf', 'Dusseldorf; Düsseldorf; F95'),
    'Greuther Furth': ('SpVgg Greuther Fürth', 'Furth; Fürth; Greuther Fürth'),
    'Hannover': ('Hannover 96', 'Hannover'),
    'Hertha': ('Hertha BSC', 'Hertha Berlin; Hertha BSC Berlin'),
    'Kaiserslautern': ('1. FC Kaiserslautern', 'FCK; Lautern'),
    'Karlsruhe': ('Karlsruher SC', 'KSC; Karlsruhe'),
    'Magdeburg': ('1. FC Magdeburg', 'Magdeburg'),
    'Nurnberg': ('1. FC Nürnberg', 'Nuremberg; Nürnberg; Nurnberg; FCN'),
    'Osnabruck': ('VfL Osnabrück', 'Osnabrück; Osnabruck'),
    # --- England
    'Man City': ('Manchester City', 'Man. City; MCFC'),
    'Man United': ('Manchester United', 'Man Utd; Man U; Manchester Utd; MUFC'),
    'Newcastle': ('Newcastle United', 'Newcastle Utd; NUFC'),
    "Nott'm Forest": ('Nottingham Forest', 'Forest; Notts Forest'),
    'Tottenham': ('Tottenham Hotspur', 'Spurs; Tottenham'),
    'West Ham': ('West Ham United', 'West Ham Utd; Hammers'),
    'Wolves': ('Wolverhampton Wanderers', 'Wolverhampton; Wolves'),
    'Brighton': ('Brighton & Hove Albion', 'Brighton and Hove Albion; Brighton Hove Albion; Seagulls'),
    'Leeds': ('Leeds United', 'Leeds Utd'),
    'Bournemouth': ('AFC Bournemouth', 'Bournemouth'),
    'Hull': ('Hull City', 'Hull'),
    'Ipswich': ('Ipswich Town', 'Ipswich'),
    'Sheffield Weds': ('Sheffield Wednesday', 'Sheff Wed; Sheffield Weds; Owls'),
    'Sheffield United': ('Sheffield United', 'Sheff Utd; Sheff United; Blades'),
    'West Brom': ('West Bromwich Albion', 'West Bromwich; WBA'),
    'QPR': ('Queens Park Rangers', 'QPR'),
    'Norwich': ('Norwich City', 'Norwich'),
    'Cardiff': ('Cardiff City', 'Cardiff'),
    'Swansea': ('Swansea City', 'Swansea'),
    'Stoke': ('Stoke City', 'Stoke'),
    'Leicester': ('Leicester City', 'Leicester'),
    'Derby': ('Derby County', 'Derby'),
    'Preston': ('Preston North End', 'Preston; PNE'),
    'Birmingham': ('Birmingham City', 'Birmingham'),
    'Blackburn': ('Blackburn Rovers', 'Blackburn'),
    'Bolton': ('Bolton Wanderers', 'Bolton'),
    'Coventry': ('Coventry City', 'Coventry'),
    'Oxford': ('Oxford United', 'Oxford Utd; Oxford'),
    'Lincoln': ('Lincoln City', 'Lincoln'),
    'Charlton': ('Charlton Athletic', 'Charlton'),
    'Wrexham': ('Wrexham AFC', 'Wrexham'),
    'Middlesbrough': ('Middlesbrough', 'Boro'),
    'Aston Villa': ('Aston Villa', 'Villa'),
    'Crystal Palace': ('Crystal Palace', 'Palace'),
    # --- Spain
    'Ath Bilbao': ('Athletic Club', 'Athletic Bilbao; Athletic; Bilbao'),
    'Ath Madrid': ('Atlético Madrid', 'Atletico Madrid; Atleti; Atl Madrid'),
    'Barcelona': ('FC Barcelona', 'Barca; Barça; FCB Barcelona'),
    'Betis': ('Real Betis', 'Betis'),
    'Celta': ('Celta Vigo', 'RC Celta; Celta de Vigo'),
    'Espanol': ('RCD Espanyol', 'Espanyol; Espanol'),
    'La Coruna': ('Deportivo La Coruña', 'Deportivo; Depor; Deportivo La Coruna'),
    'Malaga': ('Málaga CF', 'Malaga'),
    'Sociedad': ('Real Sociedad', 'Real Sociedad; La Real'),
    'Vallecano': ('Rayo Vallecano', 'Rayo; Vallecano'),
    'Villarreal': ('Villarreal CF', 'Villareal; Villarreal'),
    'Alaves': ('Deportivo Alavés', 'Alaves; Deportivo Alaves'),
    'Santander': ('Racing Santander', 'Racing de Santander; Racing'),
    'Oviedo': ('Real Oviedo', 'Oviedo'),
    'Sp Gijon': ('Sporting Gijón', 'Sporting de Gijon; Sporting Gijon; Gijon'),
    'Valladolid': ('Real Valladolid', 'Valladolid'),
    'Zaragoza': ('Real Zaragoza', 'Zaragoza'),
    'Sociedad B': ('Real Sociedad B', 'Real Sociedad II'),
    'Leganes': ('CD Leganés', 'Leganes'),
    'Mirandes': ('CD Mirandés', 'Mirandes'),
    'Almeria': ('UD Almería', 'Almeria'),
    'Castellon': ('CD Castellón', 'Castellon'),
    # --- Italy
    'Inter': ('Inter Milan', 'Inter; Internazionale; FC Internazionale; Inter Milano'),
    'Milan': ('AC Milan', 'Milan; AC Milan; Milan AC'),
    'Roma': ('AS Roma', 'Roma'),
    'Verona': ('Hellas Verona', 'Verona'),
    'Napoli': ('SSC Napoli', 'Napoli'),
    'Lazio': ('SS Lazio', 'Lazio'),
    'Juventus': ('Juventus', 'Juve'),
    'Atalanta': ('Atalanta BC', 'Atalanta'),
    'Sudtirol': ('FC Südtirol', 'Sudtirol; Südtirol; Sud Tirol'),
    'Sampdoria': ('UC Sampdoria', 'Sampdoria'),
    'Juve Stabia': ('Juve Stabia', 'Juve Stabia'),
    # --- France
    'Paris SG': ('Paris Saint-Germain', 'PSG; Paris Saint Germain; Paris SG; Paris St Germain; Paris St-Germain'),
    'Marseille': ('Olympique Marseille', 'OM; Olympique de Marseille; Marseille'),
    'Lyon': ('Olympique Lyonnais', 'OL; Olympique Lyon; Lyon'),
    'Monaco': ('AS Monaco', 'Monaco'),
    'Lille': ('LOSC Lille', 'Lille OSC; Lille'),
    'Nice': ('OGC Nice', 'Nice'),
    'Rennes': ('Stade Rennais', 'Rennes; Stade Rennes'),
    'Brest': ('Stade Brestois', 'Brest; Stade Brestois 29'),
    'Reims': ('Stade de Reims', 'Reims'),
    'St Etienne': ('AS Saint-Étienne', 'Saint Etienne; Saint-Etienne; St-Etienne; ASSE'),
    # --- Netherlands
    'Ajax': ('AFC Ajax', 'Ajax; Ajax Amsterdam'),
    'PSV Eindhoven': ('PSV', 'PSV; PSV Eindhoven'),
    'AZ Alkmaar': ('AZ Alkmaar', 'AZ'),
    'Nijmegen': ('NEC Nijmegen', 'NEC; NEC Nijmegen'),
    'For Sittard': ('Fortuna Sittard', 'Sittard'),
    'Zwolle': ('PEC Zwolle', 'PEC; Zwolle'),
    'Den Haag': ('ADO Den Haag', 'ADO; Den Haag'),
    'Heracles': ('Heracles Almelo', 'Almelo; Heracles'),
    'Twente': ('FC Twente', 'Twente'),
    'Utrecht': ('FC Utrecht', 'Utrecht'),
    'Heerenveen': ('SC Heerenveen', 'Heerenveen'),
    'Groningen': ('FC Groningen', 'Groningen'),
    # --- Portugal
    'Sp Lisbon': ('Sporting CP', 'Sporting; Sporting Lisbon; Sporting Lisboa; Sporting Clube de Portugal'),
    'Sp Braga': ('SC Braga', 'Braga; Sporting Braga; SC Braga'),
    'Porto': ('FC Porto', 'Porto'),
    'Benfica': ('SL Benfica', 'Benfica'),
    'Guimaraes': ('Vitória Guimarães', 'Vitoria Guimaraes; Vitoria SC; Guimaraes; Guimarães'),
    'Famalicao': ('FC Famalicão', 'Famalicao; Famalicão'),
    'Estoril': ('Estoril Praia', 'Estoril'),
    'Estrela': ('Estrela da Amadora', 'Estrela Amadora; Estrela'),
    'Nacional': ('CD Nacional', 'Nacional; Nacional da Madeira'),
    # --- Belgium
    'St. Gilloise': ('Union Saint-Gilloise', 'Union SG; Royal Union SG; Union St Gilloise; Union Saint Gilloise'),
    'Standard': ('Standard Liège', 'Standard Liege; Standard'),
    'Antwerp': ('Royal Antwerp', 'Antwerp; Royal Antwerp FC'),
    'Gent': ('KAA Gent', 'Gent; Ghent'),
    'Genk': ('KRC Genk', 'Genk'),
    'Anderlecht': ('RSC Anderlecht', 'Anderlecht'),
    'Mechelen': ('KV Mechelen', 'Mechelen'),
    'Kortrijk': ('KV Kortrijk', 'Kortrijk'),
    'Westerlo': ('KVC Westerlo', 'Westerlo'),
    'St Truiden': ('Sint-Truiden', 'Sint Truiden; St Truiden; STVV'),
    'Oud-Heverlee Leuven': ('OH Leuven', 'OH Leuven; Leuven; OHL'),
    'Waregem': ('Zulte Waregem', 'Zulte Waregem; Zulte-Waregem; Waregem'),
    # --- Greece
    'AEK': ('AEK Athens', 'AEK Athen; AEK Athens; AEK'),
    'Aris': ('Aris Thessaloniki', 'Aris; Aris Salonika'),
    'Olympiakos': ('Olympiacos', 'Olympiacos; Olympiakos Piraeus; Olympiakos'),
    'PAOK': ('PAOK Thessaloniki', 'PAOK; PAOK Salonika'),
    'Panathinaikos': ('Panathinaikos', 'Panathinaikos; Panathinaikos Athens'),
    # --- Scotland
    'Hearts': ('Heart of Midlothian', 'Hearts; Heart of Midlothian; Hearts FC'),
    'Hibernian': ('Hibernian', 'Hibs'),
    # --- Nordics, Switzerland, Austria
    'Malmo FF': ('Malmö FF', 'Malmo; Malmö'),
    'Hacken': ('BK Häcken', 'Hacken; Häcken; BK Hacken'),
    'Goteborg': ('IFK Göteborg', 'IFK Goteborg; Goteborg; Gothenburg'),
    'Djurgarden': ('Djurgårdens IF', 'Djurgarden; Djurgården; Djurgardens IF'),
    'Norrkoping': ('IFK Norrköping', 'IFK Norrkoping; Norrköping'),
    'Elfsborg': ('IF Elfsborg', 'Elfsborg'),
    'Bodo/Glimt': ('FK Bodø/Glimt', 'Bodo Glimt; Bodø/Glimt; Bodo-Glimt; FK Bodo Glimt'),
    'Valerenga': ('Vålerenga', 'Valerenga; Vålerenga'),
    'FC Copenhagen': ('FC København', 'Copenhagen; Kobenhavn; FC Kobenhavn; København'),
    'Brondby': ('Brøndby IF', 'Brondby; Brøndby'),
    'Young Boys': ('BSC Young Boys', 'YB; BSC Young Boys; Young Boys Bern'),
    'Zurich': ('FC Zürich', 'Zürich; Zurich; FCZ'),
    'Basel': ('FC Basel', 'Basel'),
    'Salzburg': ('Red Bull Salzburg', 'RB Salzburg; FC Salzburg; Salzburg'),
    'SK Rapid': ('Rapid Wien', 'Rapid Vienna; Rapid Wien; SK Rapid Wien'),
    'Austria Vienna': ('Austria Wien', 'FK Austria Wien; Austria Vienna; Austria Wien'),
    # --- Eastern Europe
    'Legia': ('Legia Warszawa', 'Legia Warsaw; Legia'),
    'Lech Poznan': ('Lech Poznań', 'Lech Poznan; Lech'),
    'FCSB': ('FCSB', 'Steaua Bucuresti; Steaua Bucharest; Steaua; FCSB'),
    'Zenit': ('Zenit St. Petersburg', 'Zenit St Petersburg; Zenit Saint Petersburg; Zenit'),
    'CSKA Moscow': ('CSKA Moscow', 'CSKA Moskva; CSKA'),
    'Spartak Moscow': ('Spartak Moscow', 'Spartak Moskva; Spartak'),
    'Dynamo Moscow': ('Dynamo Moscow', 'Dinamo Moscow; Dinamo Moskva'),
    # --- Americas, Japan
    'Flamengo RJ': ('Flamengo', 'Flamengo; CR Flamengo'),
    'Botafogo RJ': ('Botafogo', 'Botafogo'),
    'Atletico-MG': ('Atlético Mineiro', 'Atletico Mineiro; Atletico MG; Galo'),
    'Athletico-PR': ('Athletico Paranaense', 'Athletico Paranaense; Athletico PR; Atletico Paranaense'),
    'Sao Paulo': ('São Paulo', 'Sao Paulo; São Paulo FC'),
    'Gremio': ('Grêmio', 'Gremio; Grêmio'),
    'Bragantino': ('Red Bull Bragantino', 'Bragantino; RB Bragantino'),
    'Vasco': ('Vasco da Gama', 'Vasco; Vasco da Gama'),
    'Velez Sarsfield': ('Vélez Sarsfield', 'Velez; Vélez; Velez Sarsfield'),
    'Club America': ('Club América', 'America; América; Club America'),
    'Guadalajara Chivas': ('Chivas Guadalajara', 'Chivas; Guadalajara; CD Guadalajara'),
    'UNAM Pumas': ('Pumas UNAM', 'Pumas; UNAM'),
    'Club Tijuana': ('Xolos Tijuana', 'Tijuana; Xolos; Club Tijuana'),
    'Club Leon': ('Club León', 'Leon; León; Club Leon'),
    'Tigres UANL': ('Tigres UANL', 'Tigres'),
    'Los Angeles FC': ('Los Angeles FC', 'LAFC; LA FC'),
    'Los Angeles Galaxy': ('LA Galaxy', 'LA Galaxy; Los Angeles Galaxy; Galaxy'),
    'New York City': ('New York City FC', 'NYCFC; NYC FC; New York City'),
    'New York Red Bulls': ('New York Red Bulls', 'NY Red Bulls; Red Bulls'),
    'Atlanta Utd': ('Atlanta United', 'Atlanta United FC; Atlanta Utd'),
    'DC United': ('D.C. United', 'DC United; DC Utd'),
    'Minnesota United': ('Minnesota United', 'Minnesota Utd'),
    'Yokohama F. Marinos': ('Yokohama F. Marinos', 'Yokohama Marinos; Yokohama FM; Marinos'),
}

# --- national teams: canonical (as in the results data) -> aliases
NATIONS: dict[str, str] = {
    'Turkey': 'Türkiye; Turkiye; Turkei; Turquia',
    'Czech Republic': 'Czechia; Czech Rep; Cesko',
    'Ivory Coast': "Cote d'Ivoire; Côte d'Ivoire; Cote dIvoire",
    'South Korea': 'Korea Republic; Korea; Republic of Korea',
    'North Korea': 'Korea DPR; DPR Korea',
    'United States': 'USA; US; United States of America; USMNT',
    'Republic of Ireland': 'Ireland; Rep. of Ireland; Eire',
    'North Macedonia': 'Macedonia; FYR Macedonia',
    'Bosnia and Herzegovina': 'Bosnia; Bosnia Herzegovina; Bosnia-Herzegovina',
    'Cape Verde': 'Cabo Verde; Cape Verde Islands',
    'DR Congo': 'Congo DR; DRC; Democratic Republic of Congo; Zaire',
    'Netherlands': 'Holland; Nederland',
    'United Arab Emirates': 'UAE',
    'Curaçao': 'Curacao',
    'Eswatini': 'Swaziland',
    'Myanmar': 'Burma',
    'Timor-Leste': 'East Timor',
    'Germany': 'Deutschland',
    'Spain': 'Espana; España',
    'Switzerland': 'Schweiz; Suisse',
    'Austria': 'Osterreich; Österreich',
    'Denmark': 'Danmark',
    'Sweden': 'Sverige',
    'Norway': 'Norge',
    'Croatia': 'Hrvatska',
    'Serbia': 'Srbija',
    'Greece': 'Hellas',
    'Georgia': 'Sakartvelo',
}


# =============================================================================
# THE INDEX
# =============================================================================
@dataclass(frozen=True)
class Match:
    canonical: str
    kind: str                 # 'club' | 'national'
    how: str                  # memory | exact | alias | core | contains | fuzzy
    display: str
    league: str | None = None

    @property
    def sure(self) -> bool:
        return self.how != 'fuzzy'


def memory_path() -> str:
    from scripts import config
    base = os.environ.get('FOOTBALL_PREDICTOR_TOTO_DIR',
                          os.path.join(config.DATA_DIR, '_toto'))
    return os.path.join(base, 'name_memory.json')


def _load_memory() -> dict[str, dict]:
    try:
        with open(memory_path(), encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


class NameIndex:
    """Resolve and search club and national-team names.

    clubs     canonical club name -> league key (or any mapping/iterable of
              names; the league is only used for display)
    nations   iterable of canonical national-team names
    """

    def __init__(self, clubs, nations=()):
        self.clubs = dict(clubs) if hasattr(clubs, 'items') else {c: None for c in clubs}
        self.nations = list(nations)
        self.entries: dict[str, tuple[str, str | None]] = {}   # canonical -> (kind, league)
        self.display_of: dict[str, str] = {}
        self.aliases_of: dict[str, list[str]] = {}
        for c, lg in self.clubs.items():
            self.entries[c] = ('club', lg)
        for n in self.nations:
            self.entries.setdefault(n, ('national', None))
        for c in self.entries:
            disp, alias = TEAMS.get(c, (c, ''))
            if c in NATIONS and self.entries[c][0] == 'national':
                disp, alias = c, NATIONS[c]
            self.display_of[c] = disp
            self.aliases_of[c] = [a.strip() for a in alias.split(';') if a.strip()]
        # exact / alias / core maps: key -> set(canonical)
        self._exact: dict[str, set[str]] = {}
        self._alias: dict[str, set[str]] = {}
        self._core: dict[tuple, set[str]] = {}
        for c in self.entries:
            self._exact.setdefault(norm(c), set()).add(c)
            for name in [self.display_of[c]] + self.aliases_of[c]:
                key = norm(name)
                if key != norm(c):
                    self._alias.setdefault(key, set()).add(c)
            for name in [c, self.display_of[c]] + self.aliases_of[c]:
                self._core.setdefault(core(name), set()).add(c)
        self._norm_of = {c: norm(c) for c in self.entries}
        self._mem_mtime, self._mem = None, {}

    # ---------------------------------------------------------------- memory
    def _memory(self) -> dict:
        try:
            mt = os.path.getmtime(memory_path())
        except OSError:
            mt = None
        if mt != self._mem_mtime:
            self._mem_mtime, self._mem = mt, _load_memory()
        return self._mem

    def remember(self, typed: str, canonical: str) -> bool:
        if canonical not in self.entries or not norm(typed):
            return False
        mem = dict(_load_memory())
        mem[norm(typed)] = {'canonical': canonical}
        path = memory_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(mem, f, indent=2, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)
        self._mem_mtime = None
        return True

    def _token_clash(self, token: str, chosen: str) -> bool:
        """A one-word query ('Paris') that is also part of another team's name
        ('Paris SG') is not a safe match for the one that happened to lose its
        club-type word ('Paris FC')."""
        for k, cands in self._core.items():
            if token in k and (len(k) > 1 or cands - {chosen}):
                if cands - {chosen}:
                    return True
        return False

    # --------------------------------------------------------------- resolve
    def _make(self, canonical: str, how: str) -> Match:
        kind, league = self.entries[canonical]
        return Match(canonical, kind, how, self.display_of[canonical], league)

    def _pick(self, cands: set[str]) -> str | None:
        """One candidate, or None when ambiguous. A club beats a national team
        of the same name only when the name is literally the club's."""
        if len(cands) == 1:
            return next(iter(cands))
        return None

    def resolve(self, query: str, fuzzy: bool = True) -> Match | None:
        """The one team a name refers to, or None if it is unknown/ambiguous."""
        q = norm(query)
        if not q:
            return None
        mem = self._memory().get(q)
        if mem and mem.get('canonical') in self.entries:
            return self._make(mem['canonical'], 'memory')
        c = self._pick(self._exact.get(q, set()))
        if c:
            return self._make(c, 'exact')
        c = self._pick(self._alias.get(q, set()))
        if c:
            return self._make(c, 'alias')
        qc = core(query)
        c = self._pick(self._core.get(qc, set()))
        if c and not (len(qc) == 1 and self._token_clash(qc[0], c)):
            return self._make(c, 'core')
        # "Newcastle United" -> Newcastle: the query is exactly one known
        # name's core plus extra words, and no other name's core fits inside
        if len(qc) >= 2:
            inside = {self._pick(s) for k, s in self._core.items()
                      if len(k) < len(qc) and set(k) <= set(qc)}
            inside.discard(None)
            all_in = {x for k, s in self._core.items()
                      if len(k) < len(qc) and set(k) <= set(qc) for x in s}
            if len(inside) == 1 and len(all_in) == 1:
                return self._make(next(iter(inside)), 'contains')
        if fuzzy:
            typo = self._typo(q)
            if typo:
                return self._make(typo, 'fuzzy')
        return None

    def _tokens(self) -> set[str]:
        if not hasattr(self, '_tok'):
            self._tok = {t for c in self.entries
                         for name in [c, self.display_of[c]] + self.aliases_of[c]
                         for t in norm(name).split()}
        return self._tok

    def _typo(self, q: str) -> str | None:
        """A name that is one slip away from a known one ("buyuksehr",
        "fenerbache"): high spelling similarity and clearly the best."""
        if len(q) < 6 or q in self._tokens():       # short words / real words are
            return None                              # ambiguous, never "typos"
        best: dict[str, float] = {}
        for c in self.entries:
            r = 0.0
            for name in [c, self.display_of[c]] + self.aliases_of[c]:
                n = norm(name)
                if abs(len(n) - len(q)) <= 2:
                    r = max(r, difflib.SequenceMatcher(None, q, n).ratio())
            if r >= 0.9:
                best[c] = r
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        if not ranked:
            return None
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.05:
            return None
        return ranked[0][0]

    # --------------------------------------------------------------- suggest
    def suggest(self, query: str, limit: int = 6, min_score: float = 0.55):
        """Closest names as [(canonical, score)], best first."""
        q = norm(query)
        qc = core(query)
        if not q:
            return []
        qset = set(qc)
        scored: dict[str, float] = {}
        for c in self.entries:
            best = 0.0
            for name in [c, self.display_of[c]] + self.aliases_of[c]:
                n = norm(name)
                nc = core(name)
                if n == q or nc == qc:
                    s = 1.0
                elif qset and qset <= set(nc):
                    s = 0.9 - 0.02 * (len(nc) - len(qc))
                elif nc and set(nc) <= qset:
                    s = 0.85 - 0.02 * (len(qc) - len(nc))
                elif n.startswith(q) or q.startswith(n):
                    s = 0.85
                elif q in n or n in q:
                    s = 0.75
                else:
                    r = difflib.SequenceMatcher(None, q, n).ratio()
                    rc = difflib.SequenceMatcher(None, ' '.join(qc), ' '.join(nc)).ratio()
                    s = max(r, rc)
                    a, b = set(q.split()), set(n.split())
                    if a and b:
                        s = max(s, 0.9 * len(a & b) / len(a | b))
                best = max(best, s)
            if best >= min_score:
                scored[c] = best
        out = sorted(scored.items(), key=lambda kv: (-kv[1], len(self.display_of[kv[0]]), kv[0]))
        return out[:limit]

    # ----------------------------------------------------------- for the UI
    def search_keys(self, canonical: str) -> list[str]:
        """Normalised strings the autocomplete matches against."""
        keys = [norm(self.display_of[canonical]), norm(canonical)]
        keys += [norm(a) for a in self.aliases_of[canonical]]
        seen, out = set(), []
        for k in keys:
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def display(self, name: str) -> str:
        m = self.resolve(name, fuzzy=False)
        return m.display if m else str(name)


_INDEX_CACHE: dict = {}


def club_index(clubs) -> NameIndex:
    """A cached index over just clubs (what toto._fuzzy has to hand)."""
    key = (id(clubs), len(clubs))
    if key not in _INDEX_CACHE:
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = NameIndex(clubs)
    return _INDEX_CACHE[key]
