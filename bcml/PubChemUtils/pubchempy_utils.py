from __future__ import print_function
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from collections import Callable, defaultdict
from six.moves import xrange
try:
    # For Python 3.0 and later
    import urllib.request as urllib2
except ImportError:
    # Fall back to Python 2's urllib2
    import urllib2
import os
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
from bs4 import BeautifulSoup
from time import sleep


_base = 'pubchem.ncbi.nlm.nih.gov'
_pug_rest = '"http://pubchem.ncbi.nlm.nih.gov/pug_rest"'
_dir = os.path.dirname(__file__)
_fp_file = os.path.abspath(os.path.join(_dir, 'fingerprints.txt'))


'''
This module extends the common functionality of the PubChemPy
package
'''


class CompoundDict(OrderedDict):
    '''
    The compound dictionary is ordred and contains various levels of
    dictionaries underneath, this is the reason for the complicated structure
    '''
    def __init__(self, default_factory=defaultdict, *a, **kw):
        if (default_factory is not None and
            not isinstance(default_factory, Callable)):
                raise TypeError('First argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(tuple(self.items())))

    def __repr__(self):
        return 'CompoundDict(%s, %s)' % (self.default_factory,
                                         OrderedDict.__repr__(self))


def verbose_print(verbose, line):
    if verbose:
        print(line)


def _url_factory(uri):
    '''
    Handle the pubchem RESTful interface by passing a url directly
    '''
    uri = 'https://' + _base + uri
    print(uri)
    response = urllib2.urlopen(uri)
    value = response.read().strip()
    return value


def convert_cactvs(cactvs):
    '''
    This internal function converts 2D fingerprints to a string of 0/1s
    The fingerprint is defined here:
    ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt
    The way that this function works is:
    1) Pass cactvs
    2) Strip the 2 trailing bytes
    3) Strip the 2 leading bytes
    4) Convert the letters to base64 binary (6-bits)
    5) Report bits 32 through (881+32-11), which are the 881 informative
        bits.
    '''
    b64 = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
           "H": 7, "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13,
           "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19,
           "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25,
           "a": 26, "b": 27, "c": 28, "d": 29, "e": 30, "f": 31,
           "g": 32, "h": 33, "i": 34, "j": 35, "k": 36, "l": 37,
           "m": 38, "n": 39, "o": 40, "p": 41, "q": 42, "r": 43,
           "s": 44, "t": 45, "u": 46, "v": 47, "w": 48, "x": 49,
           "y": 50, "z": 51, "0": 52, "1": 53, "2": 54, "3": 55,
           "4": 56, "5": 57, "6": 58, "7": 59, "8": 60, "9": 61,
           "+": 62, "/": 63}
    c = cactvs[:-2].strip()
    binstring = (''.join([str(bin(b64[x]))[2:].zfill(6) for x in c.decode('utf-8')]))
    return binstring[32:-11]


def _parse_fingerprint():
    '''
    Read the NCBI fingerprint spec file and assign features to
    each fingerprint
    '''
    fp_features = {}
    with open(_fp_file) as fp:
        for line in fp:
            (pos, feature) = line.strip().split('\t')
            fp_features[int(pos)] = feature
    return fp_features


def get_binhash(cactvs):
    '''
    Convert CACTVS into a dictionary of fingerprint
    features
    '''
    fingerprint = _parse_fingerprint()
    binstring = convert_cactvs(cactvs)
    binhash = {}
    for count, val in enumerate(binstring):
        binhash[fingerprint[count]] = val
    return binhash


def cactvs_uri(ids):
    '''
    This function retreives the CACTVS uri from PubChem, which is a base64
    encoded string, specifying the 881 bits, corresponding to the
    fingerprint
    '''
    _id = str(ids)
    uri = '/rest/pug/compound/cid/' + _id + '/property/Fingerprint2D/TXT'
    return uri


def smiles_uri(ids):
    _id = str(ids)
    uri = '/rest/pug/compound/smiles/' + _id + '/cids/TXT'
    return uri


def get_smiles(_id):
    '''
    This function retreives the CID for a SMILES from PubChem
    '''
    uri = smiles_uri(_id)
    cid = _url_factory(uri)
    return cid


def stream_sdf(ids):
    '''
    This function allows bulk streaming of SDF into a data structure
    '''
    concatenated_ids = ','.join(ids)
    uri = sdf_uri(concatenated_ids)
    sdf_stream = _url_factory(uri).decode().strip('$$$$')
    sdfs = ["".join((data.lstrip(), '$$$$')) for data in
            sdf_stream.split('$$$$') if data is not ""]
    return sdfs


def sdf_uri(ids):
    '''
    This function retreives the SDF URI from PubChem
    '''
    _id = str(ids)
    uri = '/rest/pug/compound/cid/' + _id + '/record/SDF'
    return uri


def stream_xml(_id):
    '''
    This function allows streaming of pubchem XML into a data structure
    '''
    uri = xml_uri(_id)
    xml = _url_factory(uri)
    return xml


def xml_uri(_id):
    '''
    This function retreives the XML URI from PubChem
    '''
    _id = str(_id)
    uri = '/rest/pug_view/data/compound/' + _id + '/XML/'
    return uri


def extract_pubchem_xml_features(xml):
    '''
    Extracts primary PubChem Chemical and Physical data.
    If multiple values are reported
    for a given descriptor, the first is given, since, by
    convention, these are the highest quality.
    '''
    xml_glob = BeautifulSoup(xml, "lxml")
    values = {}

    def _return_value_list(text, key):
        '''Special function for returning list of values'''
        return [y.get_text() for y in text.find_next_siblings(key)]
    xml_globs = xml_glob.find_all('section')
    properties = ''
    match_text = 'Chemical and Physical Properties'
    for xml_glob in xml_globs:
        try:
            if xml_glob.find('tocheading').get_text() == match_text:
                properties = xml_glob
        except:
            pass
    try:
        for x in properties.find_all('name'):
            value = None
            name = x.get_text()
            if name not in values:
                if x.find_next_sibling('numvalue'):
                    value = x.find_next_sibling('numvalue').get_text()
                if x.find_next_sibling('stringvalue'):
                    value = x.find_next_sibling('stringvalue').get_text()
                if x.find_next_siblings('stringvaluelist'):
                    value = _return_value_list(x, 'stringvaluelist')
                if x.find_next_siblings('numvaluelist'):
                    value = _return_value_list(x, 'stringvaluelist')
                if value:
                    values[name] = value
    except:
        pass
    return values


class Collect(object):
    """Initialize variables for Collect class"""
    def __init__(self, compounds=False, fingerprint=False,
                 xml=False, sdf=False, proxy=False, user=False,
                 id_name='PubChem', chunks=False, try_count=3, verbose=False,
                 predictors=False, weights=False, smiles=False, local=False):
        self.id_name = id_name
        self.compounds = compounds
        if compounds:
            self.pubchem_ids = [x[id_name] for x in compounds]
        self.compound = CompoundDict()
        self.proxy = proxy
        self.chunks = chunks
        self.verbose = verbose
        self.smiles = smiles
        if proxy:
            self.set_proxy()
        if smiles is not False:
            id_list = []
            for count, _id in enumerate(self.pubchem_ids):
                cid = get_smiles(_id)
                id_list.append(cid)
                self.compounds[count][id_name] = cid
            self.pubchem_ids = id_list
        if predictors is not False:
            for count, _id in enumerate(self.pubchem_ids):
                self.compound[_id]['predictor'] = predictors[count]
        if weights is not False:
            for count, _id in enumerate(self.pubchem_ids):
                self.compound[_id]['weight'] = weights[count]
        self.user = user
        if user:
            self.add_user()
        self.verbose = verbose
        self.fingerprint = fingerprint
        if fingerprint:
            self.add_fingerprint(fingerprint=True)
        self.sdf = sdf
        if (sdf) and (local is False):
            self.add_sdf()
        self.xml = xml
        if xml:
            self.add_xml()
        if local:
            self.add_local(local=local)

    def set_proxy(self, proxy=False):
        """This function sets the proxy for the urllib2 library"""
        if self.proxy:
            proxy = self.proxy
        if proxy is not False:
            verbose_print(self.verbose, "Initializing proxy")
            result = urlparse(proxy)
            assert result.scheme, "Proxy must be a web address"
            proxy_support = urllib2.ProxyHandler({
                'http': proxy,
                'https': proxy
                })
            opener = urllib2.build_opener(proxy_support)
            urllib2.install_opener(opener)

    def add_user(self, user=False):
        """This function allows user features to be passed
        through the Collect Class"""
        if self.user:
            user = self.user
        if user is True:
            verbose_print(self.verbose, "Adding user provided features")
            for count, _id in enumerate(self.pubchem_ids):
                self.compound[_id]['userhash'] = self.compounds[count]['userhash']

    def add_fingerprint(self, fingerprint=False, chunks=False):
        """This function collects fingerprint data from NCBI, currently
        PubChemPy collects only ASN.1 data, which is difficult to parse
        into a binary hash of fingerprint values. It also doesn't allows
        bulk collection of the fingerprints. This function allows these"""
        if self.fingerprint:
            fingerprint = self.fingerprint
        if self.chunks:
            chunks = self.chunks
        if fingerprint is True:
            ids = self.pubchem_ids
            verbose_print(self.verbose, "Getting fingerprints from NCBI")
            fps = []
            percent = 0.
            length = float(len(self.pubchem_ids))
            if length > 100 and chunks is False:
                chunks = 100.
            if chunks is not False:
                for chunk_id in [ids[i:i + chunks] for i in xrange(0, len(ids), chunks)]:
                    '''This loop allows the ids to be chunked into size chunks. This is
                    important for really long lists, which may create problems in trying
                    to query huge numbers of ids'''
                    percent = percent + float(chunks) / length
                    #print_string = '{:2.1%} out of {}'.format(percent, length)
                    #verbose_print(self.verbose, print_string)
                    concatenated_ids = ','.join(chunk_id)
                    uri = cactvs_uri(concatenated_ids)
                    fps.extend(_url_factory(uri).splitlines())
            else:
                concatenated_ids = ','.join(ids)
                verbose_print(self.verbose, 'Collecting all fingerprints')
                uri = cactvs_uri(concatenated_ids)
                fps = _url_factory(uri).splitlines()
            for i, cactvs in enumerate(fps):
                self.compound[ids[i]]['binhash'] = get_binhash(cactvs)

    def add_local(self, local=False, chunks=False):
        directory = local
        import glob
        files = glob.glob(local + '/*.sdf')
        for filename in files:
            cpd_id = filename.split('/')[-1][:-4]
            with open(filename) as fp:
                self.compound[cpd_id]['sdf'] = fp.read()                

    def add_sdf(self, sdf=False, chunks=False):
        """This function collects NCBI sdfs and stores them for use
        in cheminformatic tools"""
        if self.sdf:
            sdf = self.sdf
        if self.chunks:
            chunks = self.chunks
        if sdf is True:
            percent = 0.
            length = float(len(self.pubchem_ids))
            ids = self.pubchem_ids
            if length > 100 and chunks is False:
                chunks = 100
            if chunks is not False:
                for chunk_id in [ids[i:i + chunks] for i in xrange(0, len(ids), chunks)]:
                    '''This loop allows the ids to be chunked into size chunks. This is
                    important for really long lists, which may create problems in trying
                    to query huge numbers of ids'''
                    percent = percent + chunks / length
                    print_string = '{:2.1%} out of {}'.format(percent, length)
                    verbose_print(self.verbose, print_string)
                    concatenated_ids = chunk_id
                    sdfs = stream_sdf(concatenated_ids)
                    for i, sdf in enumerate(sdfs):
                        self.compound[chunk_id[i]]['sdf'] = sdf
            else:
                sdfs = stream_sdf(ids)
                for i, sdf in enumerate(sdfs):
                    self.compound[ids[i]]['sdf'] = sdf

    def add_xml(self, xml=False, try_count=3):
        """This function collects NCBI XML and stores them for later parsing"""
        if self.xml:
            xml = self.xml
        if xml is True:
            percent = 0.
            length = float(len(self.pubchem_ids))
            ids = self.pubchem_ids
            verbose_print(self.verbose, 'Collecting all XMLs')
            for count, _id in enumerate(ids):
                percent = float(count) / float(length)
                #print_string = '{:2.1%} out of {}'.format(percent, length)
                #verbose_print(self.verbose, print_string)
                val = False
                count = 0
                while (val is False) and (count < try_count):
                    try:
                        xml_stream = stream_xml(_id)
                        self.compound[_id]['xml'] = extract_pubchem_xml_features(xml_stream)
                        val = True
                    except:
                        sleep(5)
                    count = count + 1
