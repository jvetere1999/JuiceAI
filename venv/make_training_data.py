import re
import pickle

import lyricsgenius

genius = lyricsgenius.Genius("vyFmFEx1VHqgVbmszAnd1SVHEDo3_tcvLywbGYCcCK0jLIWfwTq-GsctXE3av7ki")


class Gets:
    def __init__(self, artist, albums):
        self.artist = artist
        self.albums = albums

    def __call__(self):
        l = []
        for a in self.albums:
            for t in get(a, self.artist):
                s = t.song
                l.append([s.title, process_lyrics(s.lyrics)])

        return l


def process_lyrics(s):
    s = re.sub(
        r"Translations.+Lyrics|Türkçe|[0-9]+Embed|감|갑|게|고|그|극|근|기|깨|께|끝|나|너|네|년|놈|닌|다|단|닫|당|도|동랑|럼|럽|렇|로|롭|른|름|린|마|많|맺|무|미|반|방|버|빛|사|상|색|소|슈|스|싱|싸|아|우|울|움|웃|월|은|을|음|의|이|인|있|저|적|주|지|쩔|찍|차|찬|채|처|친|케|탄|파|해|햇|가|동|드|땐|란|랑|어|었|에|였|외|한|함|항|했|또",
        "", s)
    s = re.sub(r"\u2005|\u200b|\u205f|\u2060", " ", s)
    s = re.sub(r"\n", " [NEW_LINE] ", s)
    return s


def get(title, artist):
    return genius.search_album(title, artist, get_full_info=False, text_format=True).tracks


def gets(gets_list):
    l = []
    for g in gets_list:
        curr = g()
        for c in curr:
            l.append(c)

    return l


def bpe(tot):
    dict = {}
    max_key = ""
    max_value = -1
    for i in range(0, len(tot) - 2):
        if tot[i] == " " or tot[i + 1] == " ":
            continue
        temp = tot[i] + tot[i + 1]
        if temp not in dict:
            dict[temp] = 1
        else:
            dict[temp] = dict[temp] + 1
        if dict[temp] > max_value:
            max_key = temp
            max_value = dict[temp]

    print(max_key)
    tot = new_tot(tot, max_key)
    return tot


def new_tot(tot, to_replace):
    l = []
    i = 0
    while i < len(tot) - 2:
        temp = tot[i] + tot[i + 1]
        if temp == to_replace:
            l.append(to_replace)
            i = i + 1
        else:
            l.append(tot[i])
        i += 1
    return l


def make_vocab(t):
    long_str = ""
    for i in t:
        long_str = long_str + i[1]

    tot = [x for x in long_str]
    for i in range(0, 2048):
        tot = bpe(tot)

    s = sorted(set(tot))
    print(f'{len(s)} unique characters')
    print(s)
    return s


def write_file(s):
    f = open("lyrics.txt", "w")
    for i in s:
        f.write(f"[{i[0]}], [{i[1]}]\n")
    f.close()


def pull_real_data():
    jw = "Juice WRLD"

    jw_albums = ["Fighting Demons (Deluxe Edition)",
                 "Goodbye & Good Riddance (Anniversary Edition)",
                 "Death Race for Love (Bonus Track Version)",
                 "Into the Abyss (Documentary Soundtrack)",
                 "Legends Never Die",
                 "JuiceWRLD 9 9 9 (Anniversary Edition)"]
    jw_gets = Gets(jw, jw_albums)

    sb = "$UICIDEBOY$"
    sb_albums = ["Sing Me a Lullaby, My Sweet Temptation",
                 "Long Term Effects of SUFFERING",
                 "Stop Staring at the Shadows",
                 "I Want to Die in New Orleans",
                 "KILL YOUR$ELF Sagas: XVI - XX",
                 "KILL YOUR$ELF Part XVII: The $uburban $acrifice $aga"]

    sb_gets = Gets(sb, sb_albums)
    get_list = [jw_gets, sb_gets]

    a = gets(get_list)

    v = make_vocab(a)
    # Serialization
    with open("test.pickle", "wb") as outfile:
        pickle.dump(v, outfile)
    write_file(a)
    print(v)


pull_real_data()
