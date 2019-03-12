import itertools
import json
import logging
import os
import pathlib
import time

import bs4
import numpy as np
import pandas as pd
import requests


cwd = pathlib.Path.cwd()
data = cwd / 'data'

robots_txt = data / 'robots.txt'
with open(robots_txt) as fp:
    lines = iter(fp.readlines())
    for line in lines:
        if 'User-agent: *' in line:
            break

    robot_disallow = []
    for line in lines:
        if 'User-agent:' in line:
            break
        _, disallow = line.split(':')
        disallow = disallow.strip().split('*')[-1]
        robot_disallow.append(disallow)


def delay(seconds, lam=1.0):
    duration = seconds + np.random.rand() + np.random.poisson(lam=lam)
    time.sleep(duration / 4)


def get_pages(dates, *, seconds=3):
    url = 'https://www.npr.org/sections/news/archive'
    dates = pd.date_range(start, end)
    for d in dates:
        date = f'{d.month}-{d.day}-{d.year}'
        params = {'date': date}
        resp = requests.get(url, params=params)

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            logging.exception(f'HTTPError - {resp.url} - {resp.status_code}')
            continue

        page = bs4.BeautifulSoup(resp.text, 'lxml')
        delay(seconds)
        yield page


def get_info(article, *, seconds=3):
    if article.find(class_='audio-availability-message'):
        return

    teaser = article.find(class_='teaser')
    if not teaser:
        return
    date, summary = teaser.text.split('\x95')

    link = article.h2.a['href']
    if any(disallow in link for disallow in robot_disallow):
        return
    resp = requests.get(link)

    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        logging.exception(f'HTTPError - {resp.url} - {resp.status_code}')
        return

    soup = bs4.BeautifulSoup(resp.text, 'lxml')

    try:
        author = soup.find(class_='byline__name').text.strip()
    except AttributeError:
        author = None

    text = soup.find(class_='transcript')
    if not text:
        text = soup.find(id='storytext')

    try:
        paragraphs = text.find_all('p')
    except AttributeError:
        return

    story = '\n\n'.join(p.text.strip() for p in paragraphs if 'storytext' in p.parent.get('class', []))
    title = soup.find(class_='storytitle').text.strip()

    info = dict(date=date, title=title, author=author, summary=summary, story=story)
    delay(seconds)
    return info


if __name__ == '__main__':
    while True:
        try:
            start_input = input('\tEnter start date: ')
            end_input = input('\tEnter end date: ')
            dates = pd.date_range(start_input, end_input, closed='left')
        except Exception:
            print('\tInvalid date. Ctrl-C to abort.')
        else:
            break

    start = dates[0].date()
    end = dates[-1].date()

    logs = data / 'logs'
    log_file = logs / f'{start}__{end}.log'
    log_file.touch()

    jsons = data / 'jsons'
    json_file = jsons / f'{start}__{end}.json'
    json_file.touch()

    fmt = '{name} - {asctime} - {levelname} : {message}'
    logging.basicConfig(filename=log_file, level=logging.INFO, style='{', format=fmt)

    logging.info(f'STARTED {start} to {end}')

    jsons = []
    try:
        for date, page in zip(dates, get_pages(dates, seconds=0)):
            logging.info(date.date())
            for article in page.find_all('article'):
                info = get_info(article, seconds=1)
                if info:
                    jsons.append(json.dumps(info))
            with open(json_file, 'a') as fp:
                lines = (j + n for j, n in zip(jsons, itertools.repeat('\n')))
                fp.writelines(lines)
            jsons.clear()
    except Exception as exc:
        logging.exception('*** MAIN ERROR ***')
        os.system(f'say "{type(exc)} {date.date()}"')

    logging.info(f'FINISHED {start} to {end}')
