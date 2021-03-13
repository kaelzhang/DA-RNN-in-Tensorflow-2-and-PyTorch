from get_rolling_window import rolling_window


def get_labels_from_features(features, window_size, y_dim):
    return features[window_size - 1:, -y_dim:]


def split_by_ratio(features, validation_ratio):
    length = len(features)
    validation_length = int(validation_ratio * length)

    return features[:-validation_length], features[-validation_length:]


def split_data(
    data,
    apply,
    window_size,
    y_dim,
    validation_ratio
):
    train_data, val_data = split_by_ratio(data, validation_ratio)

    train_f, train_l = rolling_window(
        train_data, window_size, 1
    ), get_labels_from_features(train_data, window_size, y_dim)

    val_f, val_l = rolling_window(
        val_data, window_size, 1
    ), get_labels_from_features(val_data, window_size, y_dim)

    return apply(train_f), apply(train_l), apply(val_f), apply(val_l)
