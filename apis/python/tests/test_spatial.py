from urllib.parse import urljoin

import pyarrow as pa
import pytest

import tiledbsoma as soma


def create_and_populate_df(uri: str) -> soma.DataFrame:
    obs_arrow_schema = pa.schema(
        [
            ("foo", pa.int32()),
            ("bar", pa.float64()),
            ("baz", pa.large_string()),
        ]
    )

    with soma.DataFrame.create(uri, schema=obs_arrow_schema) as obs:
        pydict = {}
        pydict["soma_joinid"] = [0, 1, 2, 3, 4]
        pydict["foo"] = [10, 20, 30, 40, 50]
        pydict["bar"] = [4.1, 5.2, 6.3, 7.4, 8.5]
        pydict["baz"] = ["apple", "ball", "cat", "dog", "egg"]
        rb = pa.Table.from_pydict(pydict)
        obs.write(rb)

    return soma.DataFrame.open(uri)


def test_scene_basic(tmp_path):
    baseuri = tmp_path.as_uri()

    with soma.Scene.create(baseuri) as scene:
        assert scene.uri == baseuri

        with pytest.raises(TypeError):
            scene["obsl"] = soma.Experiment.create(urljoin(baseuri, "obs"))
        obsl_uri = urljoin(baseuri, "obsl")
        scene["obsl"] = soma.Collection.create(obsl_uri)
        scene["obsl"]["df"] = create_and_populate_df(urljoin(obsl_uri, "df"))

        with pytest.raises(TypeError):
            scene["varl"] = soma.Measurement.create(urljoin(baseuri, "var"))
        varl_uri = urljoin(baseuri, "varl")
        scene["varl"] = soma.Collection.create(varl_uri)
        scene["varl"]["sparse"] = soma.SparseNDArray.create(
            urljoin(varl_uri, "sparse"), type=pa.int64(), shape=(10,)
        )
        scene["varl"]["dense"] = soma.DenseNDArray.create(
            urljoin(varl_uri, "dense"), type=pa.int64(), shape=(10,)
        )

        img_uri = urljoin(baseuri, "img")
        scene["img"] = soma.Collection.create(img_uri)
        scene["img"]["col"] = soma.Collection.create(urljoin(img_uri, "col"))

    assert not soma.Collection.exists(baseuri)
    assert soma.Scene.exists(baseuri)
    assert soma.Collection.exists(obsl_uri)
    assert soma.Collection.exists(varl_uri)
    assert soma.Collection.exists(img_uri)
    assert soma.Measurement.exists(urljoin(baseuri, "var"))
    assert soma.SparseNDArray.exists(urljoin(varl_uri, "sparse"))
    assert soma.DenseNDArray.exists(urljoin(varl_uri, "dense"))
    assert soma.Collection.exists(urljoin(img_uri, "col"))

    with soma.Scene.open(baseuri) as scene:
        assert scene is not None
        assert scene.obsl is not None
        assert scene.obsl["df"] is not None
        assert scene.varl is not None
        assert scene.varl["sparse"] is not None
        assert scene.varl["dense"] is not None
        assert scene.img is not None
        assert scene.img["col"] is not None

        assert len(scene) == 3
        assert scene.soma_type == "SOMAScene"

        assert scene.obsl == scene["obsl"]
        assert len(scene.obsl) == 1
        assert scene.obsl["df"] == scene["obsl"]["df"]

        assert scene.varl == scene["varl"]
        assert len(scene.varl) == 2
        assert scene.varl["sparse"] == scene["varl"]["sparse"]
        assert scene.varl["dense"] == scene["varl"]["dense"]

        assert scene.img == scene["img"]
        assert len(scene.img) == 1
        assert scene.img["col"] == scene["img"]["col"]

    with pytest.raises(soma.DoesNotExistError):
        soma.Scene.open("bad uri")
