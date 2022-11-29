import lyricsgenius

genius = lyricsgenius.Genius("vyFmFEx1VHqgVbmszAnd1SVHEDo3_tcvLywbGYCcCK0jLIWfwTq-GsctXE3av7ki")


def get(title):
    return genius.search_album(title, "Juice WRLD")


def album_to_csv(album):
    s = ""
    for i in album.tracks:
        song = i.song
        title = song.title
        lyrics = song.lyrics.replace("\n", " ")
        s += f'{title}, {lyrics}\n'
    return s


def get_album(album):
    return album_to_csv(get(album))

#albums = ["Fighting Demons (Deluxe Edition)", "Goodbye & Good Riddance (Anniversary Edition)", "Death Race for Love (Bonus Track Version)","Into the Abyss (Documentary Soundtrack)","Legends Never Die","JuiceWRLD 9 9 9 (Anniversary Edition)"]

csv = ""
albums = ["Unreleased Songs [Discography List]"]

for album_str in albums:
    album_str = get_album(album_str)
    csv += album_str


f = open("jwleak.txt", "x")
f.write(csv)
f.close()
