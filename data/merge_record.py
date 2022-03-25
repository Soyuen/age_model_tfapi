import tensorflow as tf

shards = 10
list_of_tfrecord_files=[]
for i in range(shards):
    list_of_tfrecord_files.append(f"../imdb_tfrecords/imdb-{i}.tfrecords")
dataset = tf.data.TFRecordDataset(list_of_tfrecord_files)

# Save dataset to .tfrecord file
filename = '../imdb.tfrecords'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(dataset)
