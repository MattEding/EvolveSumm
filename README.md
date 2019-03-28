```python
>>> import dde
>>> with open('tell_tale_heart.txt') as fp:
        text = fp.read()
>>> dde_summ = DdeSummarizer(pop_size=50, max_iter=500, summ_ratio=0.05, stop_words='english',
                             n_jobs=-1, random_state=0) # 5% summary length
>>> dde_sum.fit(text)
>>> dde_sum.summarize()
>>> print(dde_summ.summary_)
Whenever it fell upon me, my blood ran cold; and so by degrees --very gradually --I made up my mind to take the life of the old man, and thus rid myself of the eye forever.
I was never kinder to the old man than during the whole week before I killed him.
And then, when I had made an opening sufficient for my head, I put in a dark lantern, all closed, closed, that no light shone out, and then I thrust in my head.
It took me an hour to place my whole head within the opening so far that I could see him as he lay upon his bed.
would a madman have been so wise as this, And then, when my head was well in the room, I undid the lantern cautiously-oh, so cautiously --cautiously (for the hinges creaked) --I undid it just so much that a single thin ray fell upon the vulture eye.
And every morning, when the day broke, I went boldly into the chamber, and spoke courageously to him, calling him by name in a hearty tone, and inquiring how he has passed the night.
His room was as black as pitch with the thick darkness, (for the shutters were close fastened, through fear of robbers,) and so I knew that he could not see the opening of the door, and I kept pushing it on steadily, steadily.
I had my head in, and was about to open the lantern, when my thumb slipped upon the tin fastening, and the old man sprang up in bed, crying out --"Who's there?"
I knew what the old man felt, and pitied him, although I chuckled at heart.
And it was the mournful influence of the unperceived shadow that caused him to feel --although he neither saw nor heard --to feel the presence of my head within the room.
When I had waited a long time, very patiently, without hearing him lie down, I resolved to open a little --a very, very little crevice in the lantern.
So I opened it --you cannot imagine how stealthily, stealthily --until, at length a simple dim ray, like the thread of the spider, shot from out the crevice and fell full upon the vulture eye.
I saw it with perfect distinctness --all a dull blue, with a hideous veil over it that chilled the very marrow in my bones; but I could see nothing else of the old man's face or person: for I had directed the ray as if by instinct, precisely upon the damned spot.
It was the beating of the old man's heart.
And now at the dead hour of the night, amid the dreadful silence of that old house, so strange a noise as this excited me to uncontrollable terror.
With a loud yell, I threw open the lantern and leaped into the room.
I then replaced the boards so cleverly, so cunningly, that no human eye --not even his --could have detected any thing wrong.
A shriek had been heard by a neighbour during the night; suspicion of foul play had been aroused; information had been lodged at the police office, and they (the officers) had been deputed to search the premises.
In the enthusiasm of my confidence, I brought chairs into the room, and desired them here to rest from their fatigues, while I myself, in the wild audacity of my perfect triumph, placed my own seat upon the very spot beneath which reposed the corpse of the victim.
But, ere long, I felt myself getting pale and wished them gone.
No doubt I now grew very pale; --but I talked more fluently, and with a heightened voice.
I arose and argued about trifles, in a high key and with violent gesticulations; but the noise steadily increased.
It grew louder --louder --louder!
I could bear those hypocritical smiles no longer!
>>> dde_summ.best_chrom_ # if you want extra control with parsing assigned clustering
array([22,  6, 18, 12, 11,  4,  9, 11, 11,  9, 15,  1, 21, 22, 16,  1,  7,
        6,  1,  3, 12, 18,  3,  0, 18, 16, 17, 23,  1,  5, 19, 14, 13, 14,
        2,  3, 13, 16, 23,  2, 20,  2,  7, 11,  8, 23,  6, 13, 11, 13, 19,
        9,  9,  9, 15, 14, 17,  0, 11, 20,  8,  6,  1,  4, 20,  9,  7,  3,
       10, 18,  9,  4, 20,  1, 12,  6,  0, 18,  9, 17, 14,  1,  8, 16,  8,
       10, 20,  6,  2, 12, 12,  4, 17,  5, 11,  1, 17,  4,  9,  9,  3, 19,
       21,  7,  0, 12,  9,  6,  1, 20,  3,  1, 16,  4,  5,  6, 15, 21, 16,
       23, 23, 21,  0, 10, 20, 20,  2, 22, 22, 11, 10,  7,  2, 10, 17, 13,
       17, 22,  8, 10, 21, 13,  4,  0,  7, 19,  6,  9,  0, 16,  8, 23, 19,
        8, 21, 11, 15, 19,  7,  4,  5,  0, 12,  6, 21,  4, 21, 21])
>>> dde_summ.n_iter_ # if using early_stopping=True you can identify the iteration count
500
```
