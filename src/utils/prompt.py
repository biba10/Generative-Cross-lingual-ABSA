import datetime
import logging


def get_table_result_string(config_string, train_test_time, f1, prec, rec):
    results = f'{config_string}\t{f1}\t{prec}\t{rec}\t{int(train_test_time)} s'
    results_head = '\tF1\tPrecision\tRecall\ttime\n' + results

    return results_head, results


def print_time_info(
        time_sec, total_examples, label, per_examples=1000, print_fce=logging.info,
        file=None
):
    formatted_time = format_time(time_sec)
    time_per_examples = time_sec / total_examples * per_examples
    print_fce(100 * '$$$$$$')
    print_fce("Times for:" + str(label))
    print_fce(f'Examples:{total_examples}')
    print_fce(f'Total time for {str(label)} format: {formatted_time}')
    print_fce(f'Total time for {str(label)} in sec: {time_sec}')
    print_fce('----')
    print_fce(f'Total time for {str(per_examples)} examples format: {format_time(time_per_examples)}')
    print_fce(f'Total time for {str(per_examples)} examples in sec: {time_per_examples}')
    print_fce('----')
    print_fce('Copy ')
    # label | per_examples | total_examples | formatted_time | time_sec | time_per_examples
    output = str(label) + '\t' + str(per_examples) + '\t' + str(total_examples) + '\t' + str(formatted_time) + \
             '\t' + str(time_sec) + '\t' + str(time_per_examples)
    print_fce(output)
    # write results to disk
    if file is not None:
        file_write = file + "_" + label + ".txt"
        with open(file_write, 'a', encoding='utf-8') as f:
            f.write(output + "\n")

    print_fce(100 * '$$$$$$')


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
