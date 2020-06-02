import numpy as np
import tinydb
from tinydb import TinyDB, Query
from tinydb.operations import delete

class RawEntry:
	def __init__(self, entry):
		self.filename = entry['filename']
		self.frame_nr = entry['frame_nr']
		self.body = np.array(entry['body'])

class BodyEntry:
	def __init__(self, entry):
		self.filename = entry['filename']
		self.frame_nr = entry['frame_nr']
		self.label = entry['label']
		self.angles = np.array(entry['angles'])

#-------- RAW --------#

def update_frame_nr(dataset, file_prefix):
	db = TinyDB("db/" + dataset + '-raw.json')
	query = Query()
	results = db.search(query.filename.matches(file_prefix))
	i = 1
	for result in results:
		print(result["filename"])
		db.update({'frame_nr': i}, query.filename == result["filename"])
		i = i + 1

def clean2():
	dataset = "pose-2"
	db = TinyDB("db/" + dataset + '-raw.json')
	entries = db.all()
	query = Query()
	print(entries)
	for entry in entries:
		db.update(delete('hand_r'), query.filename == entry['filename'])
		db.update(delete('hand_l'), query.filename == entry['filename'])

	#print(entries)

def clean():
	dataset = "pose-3"
	db = TinyDB("db/" + dataset + '-raw.json')
	query = Query()
	existing_entries = db.search(query.filename == 531)
	print(existing_entries)
	for i in range (480, 540):
		db.remove(query.filename == i)

	existing_entries = db.search(query.filename == 531)
	print(existing_entries)

def upsert_raw_entry(dataset, filename, frame_nr, body):
	db = TinyDB("db/" + dataset + '-raw.json')
	query = Query()
	existing_entries = db.search(query.filename == filename)
	if(len(existing_entries) > 1):
		print('Found duplicate entry for file ' + filename)
	db.upsert({'filename': filename, 'frame_nr': frame_nr, 'body': body.tolist()}, query.filename == filename)

def get_raw_entries(dataset, file_prefix):
	db = TinyDB("db/" + dataset + '-raw.json')
	query = Query()
	results = db.search(query.filename.matches(file_prefix))
	entries = list(map(lambda r: RawEntry(r), results))
	return entries

#-------- BODY --------#

def upsert_body_entry(dataset, filename, frame_nr, label, angles):
	db = TinyDB("db/" + dataset + '.json')
	query = Query()
	existing_entries = db.search(query.filename == filename)
	if(len(existing_entries) > 1):
		print('Found duplicate entry for file ' + filename)
	db.upsert({'filename': filename, 'frame_nr': frame_nr, 'label': label, 'angles': angles.tolist()}, query.filename == filename)


def get_body_entries(dataset, label):
	db = TinyDB("db/" + dataset + '.json')
	query = Query()
	results = db.search(query.label == label)
	entries = list(map(lambda r: BodyEntry(r), results))
	entries.sort(key=lambda x: x.frame_nr, reverse=False)
	return np.array(entries)

def get_all_body_entries(dataset):
	db = TinyDB("db/" + dataset + '.json')
	results = db.all()
	entries = np.array(list(map(lambda r: BodyEntry(r), results)))
	return np.array(entries)

#clean2()

#dataset = "pose-3"
#file_prefixes =  ["cross-arms"]
#for file_prefix in file_prefixes:
	#update_frame_nr(dataset, file_prefix)
	#entries = get_raw_entries(dataset, file_prefix)
	#for entry in entries:
		#print(entry.filename)
		#print(entry.frame_nr)

