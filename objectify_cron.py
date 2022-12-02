from contextlib import contextmanager
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import glob
import os
import random
import re
import requests
import subprocess
import smtplib
import sys

import arrow
from ics import Calendar
#from huggingface_hub import login
import openai
import torch

import keys

openai.api_key = keys.openai

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def find_newest_obj(folder_to_search='/Users/vwn277/Downloads/',no_older_than=arrow.utcnow().shift(minutes=-10)):
    newest = ''
    newest_creation_time = no_older_than
    for fname in glob.glob(folder_to_search + "*.obj"):
        created_time = arrow.get(os.stat(fname).st_mtime)
        if created_time > newest_creation_time: # then it's newer
            newest = fname
            newest_creation_time = created_time
    return newest

def pick_calendar_event():
  url = "https://calendar.google.com/calendar/ical/414dfe522ee5194cb0cc3e2256690f57303153496c3e98e5bb53730753989b9a%40group.calendar.google.com/private-9168dfcf8e330c77b23622e7655c52f2/basic.ics"
  c = Calendar(requests.get(url).text) 
  #print(c)

  #c
  # <Calendar with 118 events and 0 todo>
  #c.events
  # {<Event 'Visite de "Fab Bike"' begin:2016-06-21T15:00:00+00:00 end:2016-06-21T17:00:00+00:00>,
  # <Event 'Le lundi de l'embarqué: Adventure in Espressif Non OS SDK edition' begin:2018-02-19T17:00:00+00:00 end:2018-02-19T22:00:00+00:00>,
  #  ...}
  # e = list(c.timeline)[-1]
  # "Event '{}' started {}".format(e.name, e.begin.humanize())
  # "Event 'Workshop Git' started 2 years ago"

  now = arrow.utcnow()
  week_start = now.shift(days=+3)
  week_end = now.shift(days=+10)
  print("looking for events between {} and {}...".format(week_start, week_end))
  coming_events = list(c.timeline.overlapping(week_start, week_end))
  event_to_plan_for = random.choice(coming_events)
  return (event_to_plan_for.name, event_to_plan_for.begin)

def generate_text_prompt(base_str, str_type="event"):
    # Calendar events, self-tracking data, (my mom's birthday)
    if str_type == "event":
      return """Tell me five items I need to have before I go to {}:
  1.""".format(
          base_str
      )
    if str_type == "accomplishment":
      return """Tell me five items I can use to commemorate {}:
  1.""".format(
          base_str
      )
    if str_type == "sentiment":
      return """Tell me five items I should have when I'm feeling {}:
  1.""".format(
          base_str
      )
    return """Tell me five items anybody would love to have:
  1."""

def parse_and_pick_result(response):
    result = re.sub("[0-9]\.", "\n", response)
    result_lines = result.split("\n")
    # remove any empties
    non_empty = []
    for result_line in result_lines:
        if result_line:
            non_empty.append(result_line.strip())
    # pick a random item
    chosen = random.choice(non_empty)
    return chosen

def request_completion_from_openai(prompt):
  response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            temperature=0.6,
  )
  result = "1. " + response.choices[0].text
  return result

def create_obj(prompt):
    Prompt_text = prompt #@param {type: 'string'}
    Training_iters = 15000 #@param {type: 'integer'}
    Learning_rate = 1e-3 #@param {type: 'number'}
    Training_nerf_resolution = 64  #@param {type: 'integer'}
    # CUDA_ray = True #@param {type: 'boolean'}
    # View_dependent_prompt = True #@param {type: 'boolean'}
    # FP16 = True #@param {type: 'boolean'}
    Seed = 0 #@param {type: 'integer'}
    Lambda_entropy = 1e-4 #@param {type: 'number'}
    Max_steps = 512 #@param {type: 'number'}
    Checkpoint = 'latest' #@param {type: 'string'}
    Workspace = "bike" #@param{type: 'string'}
    # Save_mesh = True #@param {type: 'boolean'}

    # processings
    Prompt_text = "'" + Prompt_text + "'"

    # run the actual command
    torch.cuda.empty_cache()
    obj_f = ''

    with cd('stable-dreamfusion'):
        mesh_command = ['python', 'main.py', '-O2', 
                                             '--text', Prompt_text,
                                                '--workspace', Workspace, 
                                                '--iters', Training_iters,
                                                '--lr', Learning_rate,
                                                '--w', Training_nerf_resolution,
                                                '--h', Training_nerf_resolution,
                                                '--seed', Seed,
                                                '--lambda_entropy', Lambda_entropy, 
                                                '--ckpt', Checkpoint,
                                                '--save_mesh',
                                                '--max_steps', Max_steps]
        mesh_command = [str(piece) for piece in mesh_command]
        #print(' '.join(mesh_command))
        subprocess.run(mesh_command)
        output_folder = os.getcwd() + '/' + Workspace + '/mesh/'
        obj_f = glob.glob(output_folder + "*.obj")[0]

    return obj_f

def slice_mesh(obj_location):
    # PrusaSlicer/build/src/prusa-slicer --scale-to-fit 100,100,100 --load crealityplaconfig.ini --gcode bla.obj
    slice_command = ['/Users/vwn277/projects/undesign/objs/PrusaSlicer/build/src/prusa-slicer', '--scale-to-fit', '100x100x100',
                                                                            '--load', 'objs/crealityplaconfig.ini',
                                                                            '--gcode', obj_location]
    results = subprocess.check_output(slice_command)
    # the last line from Prusa Slicer is "Slicing result exported to ..."
    last_line = results.decode('utf-8').strip().split("\n")[-1]
    location = last_line.split(" exported to ")[1]

    return location

# from https://stackoverflow.com/questions/15718068/search-file-and-find-exact-match-and-print-line
def generate_lines_that_start_with(string, fp):
    for line in fp:
        if line.startswith(string):
            yield line

def move_gcode_f(gcode_f, event_time):
    # find out how long the gcode takes to print
    # we're looking for a line that says "; estimated printing time (normal mode) = 2h 42m 19s"
    new_gcode_name = gcode_f
    with open(gcode_f, "r") as f:
        default_time_line = '; estimated printing time (normal mode) = 12h'
        print_time_line = next(generate_lines_that_start_with("; estimated printing time (normal mode)",f),default_time_line)
        print_time_h = re.search('\s+([0-9]*)h',print_time_line).group(1)
        print_time_h = int(print_time_h + 1)
        event_time = event_time.shift(hours=-print_time_h)
        new_gcode_name = os.path.join(os.path.split(gcode_f)[0], event_time.format('YYYY-MM-DD HH:mm') + ".gcode")
    os.rename(gcode_f, new_gcode_name)
    return new_gcode_name

def alert_user(gcode_f):
    me = 'objectify@localhost'
    you = 'valkyrie.savage@gmail.com'

    msg = MIMEMultipart()

    msg['Subject'] = 'Objectify: I\'ve made a new object for you'
    msg['From'] = me
    msg['To'] = you
    msg.attach(MIMEText('You have a new file to print! -> %s\nMake sure you start it by the date/time in the filename.' % gcode_f, "plain"))

    # Open PDF file in binary mode
    with open(gcode_f, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {gcode_f}",
    )

    # Add attachment to message and convert message to string
    msg.attach(part)
    text = msg.as_string()

    # Send the message via our own SMTP server, but don't include the
    # envelope header.
    s = smtplib.SMTP('localhost:1025')
    s.sendmail(me, [you], text)
    s.quit()


#login(keys.huggingface)

def main():
    # chosen_event_name, chosen_event_time = pick_calendar_event()
    # print(chosen_event_name, chosen_event_time)
    # text_prompt = generate_text_prompt(chosen_event)
    # print(text_prompt)
    # text_completion = request_completion_from_openai(text_prompt)
    # print(text_completion)
    # mesh_prompt = parse_and_pick_result(text_completion)
    # print(mesh_prompt)
    # mesh_prompt = 'a bicycle'
    # obj_f = create_obj(mesh_prompt)
    #obj_f = "objs/cake.obj"
    obj_f = find_newest_obj()
    print(obj_f)
    chosen_event_time = arrow.get(os.path.split(obj_f)[1].strip('.obj'))
    gcode_f = slice_mesh(obj_f)
    print(gcode_f)
    gcode_f = move_gcode_f(gcode_f, chosen_event_time)
    print(gcode_f)
    alert_user(gcode_f)


if __name__=='__main__':
    main()