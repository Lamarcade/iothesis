import zipfile
import io
import pandas as pd


def _read_iot_file(year: int, member: str, header, table_type: str = "pxp",
                   zip_path: str = "EXIOBASE_data.zip") -> pd.DataFrame:
    """
    Read a single EXIOBASE IOT file (Z.txt, x.txt, Y.txt, ...) from the nested zips.

    Parameters
    ----------
    year : int
        Year of the IOT table (e.g. 2019).
    member : str
        Name of the file inside the inner zip (e.g. 'Z.txt').
    header : int or list of int
        Header row(s) passed to pandas.read_csv (1-row vs 2-row headers).
    table_type : str
        'pxp' (product-by-product) or 'ixi' (industry-by-industry).
    zip_path : str
        Path to the outer EXIOBASE_data.zip file.
    """
    inner_name = f"IOT_{year}_{table_type}.zip"

    with zipfile.ZipFile(zip_path) as outer:
        with outer.open(inner_name) as inner_bytes:
            with zipfile.ZipFile(io.BytesIO(inner_bytes.read())) as inner:
                with inner.open(member) as f:
                    return pd.read_csv(f, sep="\t", header=header, index_col=[0, 1])


def read_Z(year: int, table_type: str = "pxp", zip_path: str = "EXIOBASE_data.zip") -> pd.DataFrame:
    """
    Reads the Z (intermediate consumption) matrix from EXIOBASE nested zips.

    Returns
    -------
    pd.DataFrame
        Z matrix with MultiIndex (region, sector) on both rows and columns.
    """
    return _read_iot_file(year, "Z.txt", header=[0, 1], table_type=table_type, zip_path=zip_path)


def read_x(year: int, table_type: str = "pxp", zip_path: str = "EXIOBASE_data.zip") -> pd.DataFrame:
    """
    Reads the x (total output) vector from EXIOBASE nested zips.

    Returns
    -------
    pd.DataFrame
        Single column 'indout', MultiIndex (region, sector) on rows.
    """
    return _read_iot_file(year, "x.txt", header=[0], table_type=table_type, zip_path=zip_path)


def read_Y(year: int, table_type: str = "pxp", zip_path: str = "EXIOBASE_data.zip") -> pd.DataFrame:
    """
    Reads the Y (final demand) matrix from EXIOBASE nested zips.

    Returns
    -------
    pd.DataFrame
        MultiIndex (region, sector) on rows, MultiIndex (region, category) on columns.
    """
    return _read_iot_file(year, "Y.txt", header=[0, 1], table_type=table_type, zip_path=zip_path)


if __name__ == "__main__":
    Z = read_Z(year=2019, table_type="ixi")
    print(f"Shape: {Z.shape}")
    print(f"Index: {Z.index.names}, Columns: {Z.columns.names}")
    print(Z.iloc[:4, :4])
