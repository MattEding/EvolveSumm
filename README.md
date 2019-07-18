A discrete differential evolutionary (DDE) extractive text summarizer.

```
>>> from evolvesumm import DdeSummarizer

>>> with open('cask_of_amontillado.txt') as fp:
...     text = fp.read()

>>> dde_summ = DdeSummarizer(pop_size=50, summ_ratio=0.1, max_iter=100, 
...                          stop_words='english', n_jobs=-1, random_state=0)

>>> dde_summ.fit(text)

>>> dde_summ.summarize()

>>> print(dde_summ.summary_)
The thousand injuries of Fortunato I had borne as I best could; but when he ventured upon insult, I vowed revenge.
At length I would be avenged; this was a point definitely settled--but the very definitiveness with which it was resolved precluded the idea of risk.
It is equally unredressed when the avenger fails to make himself felt as such to him who has done the wrong.
For the most part their enthusiasm is adopted to suit the time and opportunity--to practise imposture upon the British and Austrian millionaires.
In painting and gemmary, Fortunato, like his countrymen, was a quack--but in the matter of old wines he was sincere.
It was about dusk, one evening during the supreme madness of the carnival season, that I encountered my friend.
But I have received a pipe of what passes for Amontillado, and I have my doubts."
said he.
Putting on a mask of black silk, and drawing a roquelaure closely about my person, I suffered him to hurry me to my palazzo.
I took from their sconces two flambeaux, and giving one to Fortunato, bowed him through several suites of rooms to the archway that led into the vaults.
"A huge human foot d'or, in a field azure; the foot crushes a serpent rampant whose fangs are embedded in the heel."
He laughed, and threw the bottle upward with a gesticulation I did not understand.
We passed through a range of low arches, descended, passed on, and, descending again, arrived at a deep crypt, in which the foulness of the air caused our flambeaux rather to glow than flame.
Its walls had been lined with human remains, piled to the vault overhead, in the fashion of the great catacombs of Paris.
Within the wall thus exposed by the displacing of the bones we perceived a still interior recess, in depth about four feet, in width three, in height six or seven.
It seemed to have been constructed for no especial use within itself, but formed merely the interval between two of the colossal supports of the roof of the catacombs, and was backed by one of their circumscribing walls of solid granite.
As for Luchesi--"
"He is an ignoramus," interrupted my friend, as he stepped unsteadily forward, while I followed immediately at his heels.
In an instant he had reached the extremity of the niche, and finding his progress arrested by the rock, stood stupidly bewildered.
I had scarcely laid the first tier of the masonry when I discovered that the intoxication of Fortunato had in a great measure worn off.
The noise lasted for several minutes, during which, that I might hearken to it with the more satisfaction, I ceased my labours and sat down upon the bones.
I again paused, and holding the flambeaux over the masonwork, threw a few feeble rays upon the figure within.
Unsheathing my rapier, I began to grope with it about the recess; but the thought of an instant reassured me.
But now there came from out the niche a low laugh that erected the hairs upon my head.
```
