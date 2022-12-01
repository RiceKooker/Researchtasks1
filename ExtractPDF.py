import pandas as pd

column_names = ['Width', 'Height', 'Thickness_mm']
file_dir = 'data/Test_specification.pkl'


def create_empty_dataframe(*args):
    df = pd.DataFrame()
    for arg in args:
        df[arg] = None
    return df


def update_df_any(df, i, **kwargs):
    for key2 in kwargs:
        key3 = key2
        df.loc[i, key3] = kwargs[key2]
        df.to_pickle(file_dir)
    return df


if __name__ == '__main__':
    # df = create_empty_dataframe(*column_names)
    # stop_asking = False
    # while not stop_asking:
    #     w = input('Width: ')
    #     h = input('Height: ')
    #     t = input('Thickness: ')
    #     df = update_df_any(df, len(df), Width=w, Height=h, Thickness_mm=t)
    #     stop = input('Do you want to stop?')
    #     if stop == 'y':
    #         break
    # print('End of entering')

    df1 = pd.read_pickle(file_dir)
    df1.to_excel("output.xlsx")

