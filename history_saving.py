def save_history(history, filepath):
    file = open(filepath, "a")
    write_metric_to_model(file, history, 'categorical_accuracy')
    write_metric_to_model(file, history, 'val_categorical_accuracy')
    write_metric_to_model(file, history, 'val_precision')
    write_metric_to_model(file, history, 'val_recall')
    write_metric_to_model(file, history, 'val_false_negative')
    write_metric_to_model(file, history, 'val_false_positive')

#writes a particular metric values to a file. In the file a '#' will precede the metric name to distinguish between substrings (metrics)
def write_metric_to_model(file, history, metric):
    file.write('#' + metric)
    print(str(history.history[metric]))
    file.write(str(history.history[metric]))
    file.write("\n")

# produces generator for possible metrics
def load_history(file_path, metrics):
    file = open(file_path, "r")

    for file_line in file:
        for metric in metrics:
            metric = '#' + metric
            if metric in file_line:
                string_with_metric_val = file_line.split('[')[1].split(']')[0]
                float_list_with_metric_val = [float(x) for x in string_with_metric_val.split(',')]
                yield float_list_with_metric_val